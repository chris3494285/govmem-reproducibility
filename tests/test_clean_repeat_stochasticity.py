from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.run_clean_repeat_stochasticity import (
    JUDGE_RECONCILIATION_NAME, LEDGER_NAME, OFFLINE_DETERMINISTIC_RESULTS_NAME, PROVIDERS, RESULTS_NAME, SEED,
    JUDGE_IDENTITY_FIELDS, _condition_rows, _prompt_hash, _resolved_observations, _canary_selection,
    build_judge_plan, build_manifest,
    build_offline_deterministic_results, default_judge_backend, dynamic_judge_requests, judge_failure_diagnostics,
    gemini_memora_transport_budget, judge_output_budget, load_frozen_lme_source, main,
    offline_deterministic_result, output_budget_record, planned_output_tokens, plan_generation,
    read_jsonl_tolerant, reconcile_judge_state, repair_manifest, select_blocks,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "results/stochasticity_control/clean_repeat_v1"
LME_SOURCE = Path("$LONGMEMEVAL_SOURCE/longmemeval-cleaned/longmemeval_oracle.json")
requires_lme_source = pytest.mark.skipif(not LME_SOURCE.exists(), reason="frozen LongMemEval native source unavailable")
CANARIES = (
    "clean-repeat-v1::openai::longmemeval::01",
    "clean-repeat-v1::anthropic::longmemeval::12",
    "clean-repeat-v1::gemini::longmemeval::21",
    "clean-repeat-v1::kimi::longmemeval::31",
    "clean-repeat-v1::xai::longmemeval::42",
    "clean-repeat-v1::deepseek::longmemeval::51",
)
FROZEN = (
    "results/formal_300_validation_v1/selected_observations.jsonl",
    "results/formal_300_validation_v1/deterministic_results.jsonl",
    "results/formal_300_validation_v1/execution_plan.jsonl",
    "results/b4_scoring_validation/private/formal_300_validation_v1/semantic_payloads.jsonl",
)


def _digests() -> dict[str, str]:
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in FROZEN}


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_selection_is_stratified_and_deterministic():
    first = select_blocks(_resolved_observations())
    second = select_blocks(_resolved_observations())
    assert [b["pilot_block_id"] for b in first] == [b["pilot_block_id"] for b in second]
    assert len(first) == 60
    assert Counter(b["source"] for b in first) == {"longmemeval": 30, "memora": 30}
    assert Counter(b["generating_provider"] for b in first) == {p: 10 for p in PROVIDERS}
    assert Counter((b["generating_provider"], b["source"]) for b in first) == {
        (p, s): 5 for p in PROVIDERS for s in ("longmemeval", "memora")
    }
    assert len({b["pilot_block_id"] for b in first}) == 60
    assert all(b["formal_score"] is not None for b in first)
    assert all(b["route"] == "semantic_judge_required" for b in first)


def test_selection_does_not_depend_on_correctness_value():
    resolved = _resolved_observations()
    flipped = []
    for row in resolved:
        if row["condition"] == "clean" and row["route"] == "semantic_judge_required" and row["formal_score"] is not None:
            flipped.append({**row, "formal_score": 0.0 if row["formal_score"] else 1.0})
        else:
            flipped.append(row)
    base = [b["pilot_block_id"] for b in select_blocks(resolved)]
    alt = [b["pilot_block_id"] for b in select_blocks(flipped)]
    assert base == alt


def test_preflight_artifacts_and_no_api_calls(tmp_path, monkeypatch, capsys):
    before = _digests()

    class _Forbidden:
        def __init__(self, *a, **k): raise AssertionError("no provider adapter during preflight")

    monkeypatch.setattr("scripts.b4_calibration_execution.OpenAIJudgeAdapter", _Forbidden)
    monkeypatch.setattr("scripts.b4_calibration_execution.AnthropicJudgeAdapter", _Forbidden)
    monkeypatch.setattr("scripts.b4_calibration_execution.GeminiJudgeAdapter", _Forbidden)

    assert main(["--phase", "preflight", "--output", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "CLEAN_REPEAT_BLOCKS=60" in output and "PAID_CALLS_PERFORMED=0" in output
    assert f"SAMPLE_SEED={SEED}" in output
    assert "SELECTION_CONDITIONED_ON_CORRECTNESS=no" in output
    assert "NEW_GENERATION_CALLS_EXPECTED=60" in output
    for name in ("manifest.jsonl", "manifest_summary.json", "preflight_report.md",
                 "expected_calls_and_cost.json", "analysis_plan.md"):
        assert (tmp_path / name).exists(), f"missing preflight artifact {name}"

    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == 60
    for row in manifest:
        assert row["clean_B_status"] == "pending"
        assert row["clean_A_correctness_joined_after_selection"] is True
        assert row["evaluator_route_expected"]
        assert row["clean_input_ref"]["condition_id"] == row["clean_condition_id"]
        assert row["clean_input_ref"]["generation_prompt_sha256"]
        assert row["clean_A_generation_ref"]["answer_sha256"]
    summary = json.loads((tmp_path / "manifest_summary.json").read_text())
    assert summary["blocks"] == 60 and summary["duplicate_formal300_blocks"] == 0
    assert summary["clean_observations_resolved"] == 300
    assert summary["api_calls_performed"] == 0
    assert _digests() == before  # frozen artifacts untouched


def test_paid_phases_are_fail_closed(tmp_path):
    _preflight(tmp_path)
    with pytest.raises(SystemExit, match="bounded request count"):
        main(["--phase", "generate", "--output", str(tmp_path)])
    with pytest.raises(SystemExit, match="requires --execute"):
        main(["--phase", "generate", "--provider", "openai", "--max-requests", "1", "--output", str(tmp_path)])
    _copy_control(tmp_path)
    with pytest.raises(SystemExit, match="requires --execute"):
        main(["--phase", "judge", "--output", str(tmp_path)])


@pytest.fixture()
def fake_backend(monkeypatch):
    calls: list[dict] = []
    import scripts.run_clean_repeat_stochasticity as mod

    def backend(provider, prompt, model, max_tokens, temperature, extra):
        calls.append({"provider": provider, "prompt": prompt, "model": model,
                      "max_tokens": max_tokens, "temperature": temperature, "extra": extra})
        return {"answer": f"fresh answer {len(calls)}", "usage": {"input_tokens": 10, "output_tokens": 3,
                                                                 "total_tokens": 13, "request_id": f"req-{len(calls)}"}}

    monkeypatch.setattr(mod, "_GENERATION_BACKEND", backend)
    return calls


def _preflight(tmp_path):
    assert main(["--phase", "preflight", "--output", str(tmp_path)]) == 0


def test_generate_dry_run_makes_zero_calls(tmp_path, fake_backend, capsys):
    _preflight(tmp_path)
    assert main(["--phase", "generate", "--provider", "all", "--max-requests-per-provider", "1",
                 "--dry-run", "--output", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert fake_backend == []
    assert "PAID_CALLS_PERFORMED=0" in output and "DRY_RUN_CALLS=0" in output
    assert output.count("WOULD_GENERATE") == 6
    assert not (tmp_path / LEDGER_NAME).exists()


def test_generate_requires_execute_and_bound_count(tmp_path, fake_backend):
    _preflight(tmp_path)
    with pytest.raises(SystemExit, match="bounded request count"):
        main(["--phase", "generate", "--provider", "openai", "--output", str(tmp_path)])
    with pytest.raises(SystemExit, match="requires --execute"):
        main(["--phase", "generate", "--provider", "openai", "--max-requests", "1", "--output", str(tmp_path)])
    assert fake_backend == []


def test_provider_filtering_and_one_per_provider_canary(tmp_path, fake_backend):
    _preflight(tmp_path)
    assert main(["--phase", "generate", "--provider", "openai", "--max-requests", "2",
                 "--execute", "--output", str(tmp_path)]) == 0
    assert len(fake_backend) == 2 and all(c["provider"] == "openai" for c in fake_backend)

    _preflight(tmp_path)
    assert main(["--phase", "generate", "--provider", "all", "--max-requests-per-provider", "1",
                 "--execute", "--output", str(tmp_path)]) == 0
    canary = [c for c in fake_backend if c["provider"] != "openai"]
    assert len(canary) == 5 and sorted(c["provider"] for c in canary) == \
        ["anthropic", "deepseek", "gemini", "kimi", "xai"]


def test_generation_reuses_frozen_prompt_and_is_idempotent(tmp_path, fake_backend):
    _preflight(tmp_path)
    manifest = {r["repeat_block_id"]: r for r in
                (json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines())}
    assert main(["--phase", "generate", "--provider", "all", "--max-requests-per-provider", "1",
                 "--execute", "--output", str(tmp_path)]) == 0
    assert len(fake_backend) == 6
    # every call reused the exact frozen CLEAN-A prompt (hash-verified), never a re-derived one
    assert all(_prompt_hash(c["prompt"]) in {m["clean_input_ref"]["generation_prompt_sha256"] for m in manifest.values()}
               for c in fake_backend)
    ledger = [json.loads(line) for line in (tmp_path / LEDGER_NAME).read_text().splitlines()]
    results = [json.loads(line) for line in (tmp_path / RESULTS_NAME).read_text().splitlines()]
    assert sum(1 for e in ledger if e["state"] == "completed") == 6 and len(results) == 6
    assert all(e["repeat_block_id"].startswith("clean-repeat-v1::") for e in ledger)
    first_completed = {e["repeat_block_id"] for e in ledger if e["state"] == "completed"}
    # rerun: the canary advances to the next pending slot per provider and never re-calls a completed one
    assert main(["--phase", "generate", "--provider", "all", "--max-requests-per-provider", "1",
                 "--execute", "--output", str(tmp_path)]) == 0
    ledger = [json.loads(line) for line in (tmp_path / LEDGER_NAME).read_text().splitlines()]
    completed_events = [e["repeat_block_id"] for e in ledger if e["state"] == "completed"]
    assert len(completed_events) == len(set(completed_events)) == 12
    second_completed = set(completed_events) - first_completed
    assert len(second_completed) == 6 and second_completed.isdisjoint(first_completed)
    # direct check: completed slots are filtered out of any future plan
    import scripts.run_clean_repeat_stochasticity as mod
    runnable, _blocked = mod.plan_generation(
        read_manifest(tmp_path), mod._condition_rows(), tuple(PROVIDERS), None, 1, set(completed_events))
    assert first_completed.isdisjoint({r["repeat_block_id"] for r in runnable})


def read_manifest(tmp_path):
    return [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]


def test_generation_does_not_touch_frozen_formal300_artifacts(tmp_path, fake_backend):
    before = _digests()
    _preflight(tmp_path)
    assert main(["--phase", "generate", "--provider", "all", "--max-requests-per-provider", "1",
                 "--execute", "--output", str(tmp_path)]) == 0
    assert _digests() == before
    # blocked (unverifiable) slots are recorded and never generated
    diag = [json.loads(line) for line in tmp_path.joinpath("generation_diagnostics.jsonl").read_text().splitlines()]
    blocked = [d for d in diag if d["state"] == "blocked_unverifiable_input"]
    assert blocked, "expected fail-closed records for unverifiable frozen inputs"
    assert all(d["reason"] == "frozen_prompt_hash_mismatch" for d in blocked)


JUDGING = CONTROL / "judging"
JUDGE_LEDGER = JUDGING / "judge_execution_ledger.jsonl"


def _copy_control(tmp_path):
    for name in ("manifest.jsonl", RESULTS_NAME):
        (tmp_path / name).write_text((CONTROL / name).read_text(), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def fake_judge_backend(monkeypatch):
    calls: list[dict] = []
    import scripts.run_clean_repeat_stochasticity as mod

    def backend(provider, model, prompt, source, max_output_tokens):
        calls.append({"provider": provider, "model": model, "source": source})
        if source == "longmemeval":
            parsed = {"decision": "correct", "confidence": "high", "short_reason": "ok",
                      "exploratory_binary": True}
        else:
            parsed = {"decisions": [], "native_fama": 1.0, "exploratory_binary": True}
        return {"raw": json.dumps(parsed), "parsed": parsed, "diagnostics": {"provider": provider}}

    monkeypatch.setattr(mod, "_JUDGE_BACKEND", backend)
    return calls


def test_generation_remains_sixty_completed():
    ledger = [json.loads(line) for line in (CONTROL / LEDGER_NAME).read_text().splitlines() if line.strip()]
    completed = [e["repeat_block_id"] for e in ledger if e["state"] == "completed"]
    assert len(completed) == len(set(completed)) == 60
    assert not [e for e in ledger if e["state"] == "failed"]
    assert len([json.loads(line) for line in (CONTROL / RESULTS_NAME).read_text().splitlines() if line.strip()]) == 60
    manifest = {r["repeat_block_id"] for r in read_manifest(CONTROL)}
    diag = [json.loads(line) for line in (CONTROL / "generation_diagnostics.jsonl").read_text().splitlines() if line.strip()]
    # any historical blocked record must belong to a row that the repair replaced
    manifest_conditions = {r["clean_condition_id"] for r in read_manifest(CONTROL)}
    assert not [d for d in diag if d["state"] == "blocked_unverifiable_input"
                and d["clean_condition_id"] in manifest_conditions]


def _judge_inputs(tmp_path):
    _copy_control(tmp_path)
    manifest = _lines(tmp_path / "manifest.jsonl")
    results = {r["repeat_block_id"]: r for r in _lines(tmp_path / RESULTS_NAME)}
    payloads = {r["observation_id"]: r for r in _lines(
        ROOT / "results/b4_scoring_validation/private/formal_300_validation_v1/semantic_payloads.jsonl")}
    return manifest, results, payloads


def test_static_paid_judge_plan_is_route_consistent_and_eighty_eight(tmp_path):
    _copy_control(tmp_path)
    manifest = _lines(tmp_path / "manifest.jsonl")
    results = {r["repeat_block_id"]: r for r in _lines(tmp_path / RESULTS_NAME)}
    payloads = {r["observation_id"]: r for r in _lines(
        ROOT / "results/b4_scoring_validation/private/formal_300_validation_v1/semantic_payloads.jsonl")}
    plan = build_judge_plan(manifest, results, payloads)
    by_slot = Counter(r["judge_slot"] for r in plan)
    # Expectations are driven by the frozen manifest truth, not hard-coded constants.
    lme = [b for b in manifest if b["source"] == "longmemeval"]
    semantic = [b for b in lme if b["clean_A_route"] == "semantic_judge_required"]
    deterministic = [b for b in lme if b["clean_A_route"] == "deterministic_abstention"]
    memora = [b for b in manifest if b["source"] == "memora"]
    assert len(lme) == 30 and len(semantic) == 29 and len(deterministic) == 1
    assert len(memora) == 30
    # Paid semantic slots exist only for the semantic-route CLEAN-B observations.
    expected_slots = {
        "lme_primary_openai": len(semantic),
        "lme_primary_anthropic": len(semantic),
        "memora_primary_gemini": len(memora),
    }
    assert by_slot == expected_slots
    assert len(plan) == 2 * len(semantic) + len(memora) == 88
    assert len({r["judge_request_id"] for r in plan}) == 88
    assert {r["phase"] for r in plan} == {"primary"}
    # no paid request refers to the deterministic-abstention CLEAN-B block
    assert all(r["repeat_block_id"] != deterministic[0]["repeat_block_id"] for r in plan)
    # every semantic LongMemEval block has both primaries on the same observation
    for block in {r["repeat_block_id"] for r in plan if r["source"] == "longmemeval"}:
        slots = {r["judge_slot"] for r in plan if r["repeat_block_id"] == block}
        assert slots == {"lme_primary_openai", "lme_primary_anthropic"}


@requires_lme_source
def test_deterministic_abstention_clean_b_is_scored_offline(tmp_path):
    manifest, results, payloads = _judge_inputs(tmp_path)
    block = next(b for b in manifest if b["clean_A_route"] == "deterministic_abstention")
    assert block["repeat_block_id"] == "clean-repeat-v1::xai::longmemeval::43"
    record = offline_deterministic_result(block, results, load_frozen_lme_source())
    assert record["repeat_block_id"] == block["repeat_block_id"]
    assert record["formal300_block_id"] == block["formal300_block_id"]
    assert record["clean_A_observation_id"] == block["clean_A_observation_id"]
    assert record["clean_A_route"] == "deterministic_abstention"
    assert record["clean_B_route"] == "deterministic_abstention"
    assert record["clean_B_route_matches_clean_A_route"] is True
    assert record["paid_call"] is False
    # the frozen route maps the canonical abstention into a binary correctness value
    assert record["clean_B_decision"] in {"abstention_correct", "abstention_incorrect"}
    assert record["clean_B_binary_correct"] is (record["clean_B_decision"] == "abstention_correct")
    assert record["clean_B_correctness"] == float(record["clean_B_binary_correct"])
    # it reuses the frozen implementation and the frozen CLEAN-A artifact, not the obsolete scorer
    assert record["scorer_implementation"].endswith("route_longmemeval")
    assert record["clean_A_frozen_deterministic_row"]["evaluator_type"] == "native_deterministic"
    assert "obsolete" not in json.dumps(record).lower()


def test_no_paid_request_exists_for_the_deterministic_block(tmp_path):
    manifest, results, payloads = _judge_inputs(tmp_path)
    plan = build_judge_plan(manifest, results, payloads)
    assert plan  # sanity
    assert not [r for r in plan if r["repeat_block_id"] == "clean-repeat-v1::xai::longmemeval::43"]


def test_route_consistency_holds_for_all_sixty_blocks(tmp_path):
    manifest, results, payloads = _judge_inputs(tmp_path)
    plan = build_judge_plan(manifest, results, payloads)
    paid_blocks = {r["repeat_block_id"] for r in plan}
    for block in manifest:
        rid = block["repeat_block_id"]
        if block["source"] == "memora":
            # Memora has a single criterion-level route class for every CLEAN-B block
            assert rid in paid_blocks
        elif block["clean_A_route"] == "semantic_judge_required":
            assert rid in paid_blocks  # CLEAN-B keeps the semantic route
        else:
            assert rid not in paid_blocks  # CLEAN-B keeps the deterministic route (offline)
    # the final paired sample is still the frozen 60 blocks
    assert len(manifest) == 60
    assert len({b["repeat_block_id"] for b in manifest}) == 60


def test_judge_dry_run_makes_zero_calls(tmp_path, fake_judge_backend, capsys):
    _copy_control(tmp_path)
    assert main(["--phase", "judge", "--dry-run", "--output", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert fake_judge_backend == []
    for key in ("JUDGE_STATIC_TOTAL=88", "JUDGE_COMPLETED_TOTAL=0", "JUDGE_PENDING_TOTAL=88",
                "JUDGE_SELECTED_REQUESTS=88", "SELECTED_OPENAI=29", "SELECTED_ANTHROPIC=29",
                "SELECTED_GEMINI=30", "DYNAMIC_ESCALATIONS_CURRENTLY_REQUIRED=0",
                "OFFLINE_DETERMINISTIC_BLOCKS=1",
                "DRY_RUN_CALLS=0", "PAID_CALLS_PERFORMED=0"):
        assert key in output
    judging = tmp_path / "judging"
    assert (judging / "judge_execution_plan.jsonl").exists()
    assert (judging / OFFLINE_DETERMINISTIC_RESULTS_NAME).exists()
    assert not (judging / "judge_execution_ledger.jsonl").exists()
    assert not (judging / "judge_results.jsonl").exists()
    offline = _lines(judging / OFFLINE_DETERMINISTIC_RESULTS_NAME)
    assert len(offline) == 1 and offline[0]["paid_call"] is False
    assert offline[0]["repeat_block_id"] == "clean-repeat-v1::xai::longmemeval::43"


def test_judge_canary_selects_three_and_shares_lme_observation(tmp_path, fake_judge_backend, capsys):
    _copy_control(tmp_path)
    assert main(["--phase", "judge", "--canary-base", "--dry-run", "--output", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "JUDGE_SELECTED_REQUESTS=3" in output
    assert "SELECTED_OPENAI=1" in output and "SELECTED_ANTHROPIC=1" in output and "SELECTED_GEMINI=1" in output
    assert fake_judge_backend == []
    manifest = _lines(tmp_path / "manifest.jsonl")
    results = {r["repeat_block_id"]: r for r in _lines(tmp_path / RESULTS_NAME)}
    payloads = {r["observation_id"]: r for r in _lines(
        ROOT / "results/b4_scoring_validation/private/formal_300_validation_v1/semantic_payloads.jsonl")}
    selected = _canary_selection(build_judge_plan(manifest, results, payloads))
    lme = [r for r in selected if r["source"] == "longmemeval"]
    memora = [r for r in selected if r["source"] == "memora"]
    assert len(lme) == 2 and lme[0]["repeat_block_id"] == lme[1]["repeat_block_id"]
    assert {r["provider"] for r in lme} == {"openai", "anthropic"}
    # the paid LME canary is a semantic-route block, never the deterministic-abstention block
    assert lme[0]["repeat_block_id"] != "clean-repeat-v1::xai::longmemeval::43"
    assert lme[0]["clean_A_route"] == "semantic_judge_required"
    assert len(memora) == 1 and memora[0]["provider"] == "gemini"
    assert {r["provider"] for r in selected} == {"openai", "anthropic", "gemini"}


def test_judge_paid_execution_fail_closed(tmp_path, fake_judge_backend):
    _copy_control(tmp_path)
    with pytest.raises(SystemExit, match="requires --execute"):
        main(["--phase", "judge", "--max-requests", "1", "--output", str(tmp_path)])
    with pytest.raises(SystemExit, match="requires --confirm-paid-judge-calls"):
        main(["--phase", "judge", "--execute", "--max-requests", "1", "--output", str(tmp_path)])
    with pytest.raises(SystemExit, match="bounded request count"):
        main(["--phase", "judge", "--execute", "--confirm-paid-judge-calls", "--output", str(tmp_path)])
    assert fake_judge_backend == []


def test_judge_completed_request_never_duplicated_and_generations_immutable(tmp_path, fake_judge_backend, capsys):
    before_gen = (CONTROL / RESULTS_NAME).read_text()
    frozen_before = _digests()
    _copy_control(tmp_path)
    assert main(["--phase", "judge", "--max-requests", "1", "--execute", "--confirm-paid-judge-calls",
                 "--output", str(tmp_path)]) == 0
    assert len(fake_judge_backend) == 1
    ledger = _lines(tmp_path / "judging" / "judge_execution_ledger.jsonl")
    first_id = [e["judge_request_id"] for e in ledger if e["state"] == "completed"][0]
    # rerun: a new pending request is selected; the completed one is never paid twice
    assert main(["--phase", "judge", "--max-requests", "1", "--execute", "--confirm-paid-judge-calls",
                 "--output", str(tmp_path)]) == 0
    assert len(fake_judge_backend) == 2
    ledger = _lines(tmp_path / "judging" / "judge_execution_ledger.jsonl")
    done = [e["judge_request_id"] for e in ledger if e["state"] == "completed"]
    assert len(done) == len(set(done)) == 2 and first_id in done
    # CLEAN-A/CLEAN-B generations and frozen formal-300 artifacts are untouched
    assert (CONTROL / RESULTS_NAME).read_text() == before_gen
    assert _digests() == frozen_before


def test_judge_started_without_result_fails_closed(tmp_path, fake_judge_backend):
    _copy_control(tmp_path)
    judging = tmp_path / "judging"
    judging.mkdir(parents=True, exist_ok=True)
    (judging / "judge_execution_ledger.jsonl").write_text(json.dumps(
        {"judge_request_id": "clean-repeat-judge::orphan", "state": "started", "attempt": 1}) + "\n",
        encoding="utf-8")
    with pytest.raises(SystemExit, match="fail-closed"):
        main(["--phase", "judge", "--max-requests", "1", "--execute", "--confirm-paid-judge-calls",
              "--output", str(tmp_path)])
    assert fake_judge_backend == []


def test_judge_provider_filter(tmp_path, fake_judge_backend, capsys):
    _copy_control(tmp_path)
    assert main(["--phase", "judge", "--judge-provider", "gemini", "--dry-run", "--output", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "SELECTED_GEMINI=30" in output
    assert "SELECTED_OPENAI=0" in output and "SELECTED_ANTHROPIC=0" in output
    assert fake_judge_backend == []


def test_judge_execution_writes_durable_artifacts_without_hidden_reasoning(tmp_path, fake_judge_backend):
    _copy_control(tmp_path)
    assert main(["--phase", "judge", "--max-requests", "1", "--execute", "--confirm-paid-judge-calls",
                 "--output", str(tmp_path)]) == 0
    results = _lines(tmp_path / "judging" / "judge_results.jsonl")
    assert len(results) == 1
    row = results[0]
    for forbidden in ("prompt", "response_text", "raw", "reasoning"):
        assert forbidden not in row
    assert row["parsed_result"] and row["response_hash"] and row["judge_request_id"]
    diagnostics = _lines(tmp_path / "judging" / "judge_diagnostics.jsonl")
    assert diagnostics and diagnostics[0]["state"] == "completed"
    ledger = _lines(tmp_path / "judging" / "judge_execution_ledger.jsonl")
    assert [e["state"] for e in ledger] == ["started", "completed"]


def test_dynamic_escalations_are_not_pre_run_and_appear_only_when_required(tmp_path):
    _copy_control(tmp_path)
    manifest = _lines(tmp_path / "manifest.jsonl")
    results = {r["repeat_block_id"]: r for r in _lines(tmp_path / RESULTS_NAME)}
    payloads = {r["observation_id"]: r for r in _lines(
        ROOT / "results/b4_scoring_validation/private/formal_300_validation_v1/semantic_payloads.jsonl")}
    plan = build_judge_plan(manifest, results, payloads)
    assert dynamic_judge_requests(plan, {}) == []  # nothing is pre-run

    lme_rows = {r["judge_slot"]: r for r in plan if r["repeat_block_id"] ==
                min(r["repeat_block_id"] for r in plan if r["source"] == "longmemeval")}
    lme_openai = lme_rows["lme_primary_openai"]["judge_request_id"]
    lme_anthropic = lme_rows["lme_primary_anthropic"]["judge_request_id"]
    agreed = {lme_openai: {"parsed_result": {"exploratory_binary": True}},
              lme_anthropic: {"parsed_result": {"exploratory_binary": True}}}
    assert dynamic_judge_requests(plan, agreed) == []
    disagreed = {lme_openai: {"parsed_result": {"exploratory_binary": True}},
                 lme_anthropic: {"parsed_result": {"exploratory_binary": None}}}
    escalations = dynamic_judge_requests(plan, disagreed)
    assert [r["judge_slot"] for r in escalations] == ["lme_escalation_gemini"]
    assert escalations[0]["provider"] == "gemini" and escalations[0]["phase"] == "escalation"

    memora_primary = next(r for r in plan if r["repeat_block_id"] ==
                          min(r["repeat_block_id"] for r in plan if r["source"] == "memora"))
    risky = {memora_primary["judge_request_id"]: {"parsed_result": {"decisions": [
        {"criterion_id": "c1", "decision": "pass", "confidence": "low"}]}}}
    secondary = dynamic_judge_requests(plan, risky)
    assert [r["judge_slot"] for r in secondary] == ["memora_dynamic_secondary_openai"]


def test_repaired_manifest_shape_and_verifiability():
    manifest = read_manifest(CONTROL)
    assert len(manifest) == 60
    assert Counter(r["source"] for r in manifest) == {"longmemeval": 30, "memora": 30}
    assert Counter(r["generating_provider"] for r in manifest) == {p: 10 for p in PROVIDERS}
    assert Counter((r["generating_provider"], r["source"]) for r in manifest) == {
        (p, s): 5 for p in PROVIDERS for s in ("longmemeval", "memora")}
    assert all(r["clean_B_input_verifiable"] is True for r in manifest)
    assert len({r["formal300_block_id"] for r in manifest}) == 60
    summary = json.loads((CONTROL / "manifest_summary.json").read_text())
    assert summary["clean_B_input_verifiable"] == 60
    assert summary["clean_B_input_blocked_unverifiable"] == 0


def test_completed_canaries_preserved_and_pending_counts():
    manifest = {r["repeat_block_id"]: r for r in read_manifest(CONTROL)}
    ledger = [json.loads(line) for line in (CONTROL / LEDGER_NAME).read_text().splitlines() if line.strip()]
    completed = {e["repeat_block_id"] for e in ledger if e["state"] == "completed"}
    # Read the canary identities from the released audit so the check holds for
    # both the original ids and the Tier-B pseudonymous release.
    canaries = tuple(json.loads((CONTROL / "manifest_replacement_audit.json").read_text())["completed_canaries_preserved"])
    assert len(canaries) == len(CANARIES) == 6
    assert set(canaries).issubset(completed)
    assert len(completed) == 60
    for rid in canaries:
        assert rid in manifest and manifest[rid]["clean_B_input_verifiable"] is True
        assert manifest[rid]["clean_A_route"] == "semantic_judge_required"
    runnable, blocked = plan_generation(
        read_manifest(CONTROL), _condition_rows(), tuple(PROVIDERS), None, None, completed)
    assert len(blocked) == 0
    assert runnable == []  # CLEAN-B generation is complete


def test_replacement_audit_records_provenance():
    audit = json.loads((CONTROL / "manifest_replacement_audit.json").read_text())
    assert audit["selection_conditioned_on_correctness"] is False
    assert len(audit["replacements"]) == 6
    # The released artifact may carry Tier-B pseudonyms instead of the original
    # source-derived identifiers, so compare structurally: the same six unique
    # canaries are preserved and none of them was replaced.
    preserved = audit["completed_canaries_preserved"]
    assert len(preserved) == len(CANARIES) == len(set(preserved)) == 6
    manifest = read_manifest(CONTROL)
    replaced_ids = {a["original_repeat_block_id"] for a in audit["replacements"]}
    assert replaced_ids.isdisjoint(set(preserved))
    for entry in audit["replacements"]:
        assert entry["correctness_joined_after_selection"] is True
        assert entry["provider"] in PROVIDERS and entry["source"] == "longmemeval"
        assert entry["block_reason"] == "frozen_prompt_hash_mismatch"
        row = next(r for r in manifest if r["repeat_block_id"] == entry["original_repeat_block_id"])
        assert row["formal300_block_id"] == entry["replacement_formal300_block_id"]
        assert row["clean_B_input_verifiable"] is True
    originals = {a["original_formal300_block_id"] for a in audit["replacements"]}
    assert originals.isdisjoint({r["formal300_block_id"] for r in manifest})


def test_repair_is_deterministic_and_correctness_blind(tmp_path, fake_backend, monkeypatch):
    before = _digests()
    resolved = _resolved_observations()
    original = build_manifest(select_blocks(resolved))
    assert any(not r["clean_B_input_verifiable"] for r in original)  # pre-repair state has blocked rows
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in original), encoding="utf-8")
    import scripts.run_clean_repeat_stochasticity as mod
    first = mod.repair_manifest(tmp_path, resolved)["audit"]
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in original), encoding="utf-8")
    flipped = [{**r, "formal_score": (0.0 if r["formal_score"] else 1.0)}
               if r["condition"] == "clean" and r["formal_score"] is not None else r for r in resolved]
    second = mod.repair_manifest(tmp_path, flipped)["audit"]
    assert first == second and len(first) == 6
    assert fake_backend == []  # repair never generates
    assert _digests() == before  # frozen formal-300 artifacts untouched


# ---------------------------------------------------------------------------
# Judge ledger reconciliation, safe recovery, and diagnostics
# ---------------------------------------------------------------------------

def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _copy_control_with_judging(tmp_path):
    _copy_control(tmp_path)
    src = CONTROL / "judging"
    dst = tmp_path / "judging"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("judge_execution_ledger.jsonl", "judge_results.jsonl", "judge_diagnostics.jsonl",
                 "dynamic_judge_plan.jsonl", OFFLINE_DETERMINISTIC_RESULTS_NAME):
        path = src / name
        if path.exists():
            (dst / name).write_text(path.read_text(), encoding="utf-8")
    return tmp_path


def _event(rid: str, state: str, attempt: int = 1, **extra) -> dict:
    return {"judge_request_id": rid, "state": state, "attempt": attempt, **extra}


def _result(rid: str) -> dict:
    return {"judge_request_id": rid, "parsed_result": {"exploratory_binary": True}}


def test_reconcile_started_then_completed():
    state = reconcile_judge_state({"r1"}, [_event("r1", "started"), _event("r1", "completed")], [_result("r1")])
    entry = state["requests"]["r1"]
    assert entry["status"] == "completed_with_result"
    assert not entry["fail_closed"] and not entry["recoverable"]


def test_reconcile_started_then_failed_is_terminal_not_interrupted():
    state = reconcile_judge_state({"r1"}, [_event("r1", "started"), _event("r1", "failed")], [])
    entry = state["requests"]["r1"]
    assert entry["status"] == "failed"
    assert entry["recoverable"] and not entry["fail_closed"]
    assert state["recoverable"] == ["r1"]


def test_reconcile_started_without_terminal_is_fail_closed():
    state = reconcile_judge_state({"r1"}, [_event("r1", "started")], [])
    entry = state["requests"]["r1"]
    assert entry["status"] == "started_without_terminal"
    assert entry["fail_closed"] and not entry["recoverable"]
    assert state["fail_closed"] == ["r1"]


def test_reconcile_duplicate_completed_events_conflict():
    events = [_event("r1", "started"), _event("r1", "completed"), _event("r1", "completed")]
    state = reconcile_judge_state({"r1"}, events, [_result("r1")])
    entry = state["requests"]["r1"]
    assert entry["status"] == "duplicate_completed_events" and entry["fail_closed"]


def test_reconcile_completed_event_without_durable_result():
    state = reconcile_judge_state({"r1"}, [_event("r1", "started"), _event("r1", "completed")], [])
    entry = state["requests"]["r1"]
    assert entry["status"] == "completed_without_result" and entry["fail_closed"]


def test_reconcile_failed_then_recovered_on_attempt_two():
    events = [_event("r1", "started", 1), _event("r1", "failed", 1),
              _event("r1", "started", 2), _event("r1", "completed", 2)]
    state = reconcile_judge_state({"r1"}, events, [_result("r1")])
    entry = state["requests"]["r1"]
    assert entry["status"] == "completed_with_result"
    assert entry["attempts"] == 2 and entry["recovered"]
    assert not entry["recoverable"] and not entry["fail_closed"]


def test_reconcile_result_without_terminal_is_conflict():
    state = reconcile_judge_state({"r1"}, [], [_result("r1")])
    assert state["requests"]["r1"]["status"] == "result_without_terminal"
    assert state["requests"]["r1"]["fail_closed"]


def test_reconcile_identity_drift_is_conflict():
    events = [_event("r1", "started", provider="openai"), _event("r1", "failed", provider="gemini")]
    state = reconcile_judge_state({"r1"}, events, [])
    assert state["requests"]["r1"]["status"] == "conflicting_identity"
    assert state["requests"]["r1"]["fail_closed"]


def test_reconcile_unattempted_plan_rows_are_pending():
    state = reconcile_judge_state({"r1", "r2"}, [_event("r1", "started"), _event("r1", "completed")],
                                  [_result("r1")])
    assert state["requests"]["r2"]["status"] == "unattempted"
    assert not state["requests"]["r2"]["fail_closed"]


def _synthetic_failed_requests(tmp_path, count=1, provider="gemini", slot="memora_primary_gemini"):
    """Tmp control dir (manifest + results) whose ledger has `count` failed requests.

    Recovery tests must not depend on the live ledger still containing failures
    (the completed experiment has none), so a controlled failed state is built
    from real frozen plan identities.
    """
    _copy_control(tmp_path)
    manifest = _lines(tmp_path / "manifest.jsonl")
    results = {r["repeat_block_id"]: r for r in _lines(tmp_path / RESULTS_NAME)}
    payloads = {r["observation_id"]: r for r in _lines(
        ROOT / "results/b4_scoring_validation/private/formal_300_validation_v1/semantic_payloads.jsonl")}
    plan = build_judge_plan(manifest, results, payloads)
    rows = [r for r in plan if r["provider"] == provider and r["judge_slot"] == slot][:count]
    assert len(rows) == count
    judging = tmp_path / "judging"
    judging.mkdir(parents=True, exist_ok=True)
    events = []
    for row in rows:
        identity = {k: row[k] for k in JUDGE_IDENTITY_FIELDS}
        events.append({**identity, "state": "started", "attempt": 1})
        events.append({**identity, "state": "failed", "attempt": 1,
                       "error_class": "ValueError", "error_message": "malformed JSON"})
    (judging / "judge_execution_ledger.jsonl").write_text(
        "".join(json.dumps(e, sort_keys=True) + "\n" for e in events), encoding="utf-8")
    return rows


def test_reconciled_real_ledger_matches_known_state(tmp_path):
    _copy_control_with_judging(tmp_path)
    ledger, _ = read_jsonl_tolerant(tmp_path / "judging" / "judge_execution_ledger.jsonl")
    results, _ = read_jsonl_tolerant(tmp_path / "judging" / "judge_results.jsonl")
    state = reconcile_judge_state(set(), ledger, results)
    # expectations are derived from the ledger itself, so the test stays valid
    # whether or not outstanding failures remain
    completed_ids = {e["judge_request_id"] for e in ledger if e["state"] == "completed"}
    result_ids = {r["judge_request_id"] for r in results}
    failed_ids = {e["judge_request_id"] for e in ledger if e["state"] == "failed"}
    assert state["counts"]["completed_with_result"] == len(completed_ids & result_ids)
    assert state["counts"].get("failed", 0) == len(failed_ids - completed_ids)
    assert completed_ids
    assert state["fail_closed"] == []
    assert set(state["recoverable"]) == (failed_ids - completed_ids)


def test_recovery_requires_execute_confirmation_and_bound(tmp_path, fake_judge_backend):
    _synthetic_failed_requests(tmp_path, 1)
    with pytest.raises(SystemExit, match="requires --execute"):
        main(["--phase", "judge", "--recover-failed", "--max-requests", "1", "--output", str(tmp_path)])
    with pytest.raises(SystemExit, match="requires --confirm-paid-judge-calls"):
        main(["--phase", "judge", "--recover-failed", "--execute", "--max-requests", "1", "--output", str(tmp_path)])
    with pytest.raises(SystemExit, match="bounded request count"):
        main(["--phase", "judge", "--recover-failed", "--execute", "--confirm-paid-judge-calls",
              "--output", str(tmp_path)])
    assert fake_judge_backend == []


def test_normal_judge_does_not_reselect_recorded_requests(tmp_path, fake_judge_backend, capsys):
    _synthetic_failed_requests(tmp_path, 2)
    ledger = _read_lines(tmp_path / "judging" / "judge_execution_ledger.jsonl")
    recorded = {e["judge_request_id"] for e in ledger}
    completed = {e["judge_request_id"] for e in ledger if e["state"] == "completed"}
    assert main(["--phase", "judge", "--dry-run", "--output", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    selected = {line.split()[1] for line in out.splitlines() if line.startswith("WOULD_JUDGE")}
    assert (recorded - completed).isdisjoint(selected)  # failed/interrupted never re-selected
    assert completed.isdisjoint(selected)
    assert fake_judge_backend == []


def test_recovery_targets_only_failed_and_is_bounded(tmp_path, fake_judge_backend, capsys):
    _synthetic_failed_requests(tmp_path, 3)
    ledger_before = _read_lines(tmp_path / "judging" / "judge_execution_ledger.jsonl")
    failed = {e["judge_request_id"] for e in ledger_before if e["state"] == "failed"}
    assert len(failed) == 3
    assert main(["--phase", "judge", "--recover-failed", "--max-requests", "1", "--dry-run",
                 "--output", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "JUDGE_RECOVERY_MODE=yes" in out and "JUDGE_SELECTED_REQUESTS=1" in out
    selected = [line.split()[1] for line in out.splitlines() if line.startswith("WOULD_JUDGE")]
    assert len(selected) == 1 and selected[0] in failed
    assert fake_judge_backend == []
    assert main(["--phase", "judge", "--recover-failed", "--max-requests", "2", "--execute",
                 "--confirm-paid-judge-calls", "--output", str(tmp_path)]) == 0
    assert len(fake_judge_backend) == 2


def test_recovery_records_next_attempt_and_preserves_history(tmp_path, fake_judge_backend):
    _synthetic_failed_requests(tmp_path, 1)
    ledger_path = tmp_path / "judging" / "judge_execution_ledger.jsonl"
    before = _read_lines(ledger_path)
    failed = {e["judge_request_id"] for e in before if e["state"] == "failed"}
    max_attempt = {rid: max(e.get("attempt", 1) for e in before if e["judge_request_id"] == rid) for rid in failed}
    assert main(["--phase", "judge", "--recover-failed", "--max-requests", "1", "--execute",
                 "--confirm-paid-judge-calls", "--output", str(tmp_path)]) == 0
    after = _read_lines(ledger_path)
    assert after[:len(before)] == before  # historical events preserved, append-only
    recovered = [e for e in after[len(before):] if e["state"] == "completed"]
    assert len(recovered) == 1 and recovered[0]["judge_request_id"] in failed
    rid = recovered[0]["judge_request_id"]
    assert recovered[0]["attempt"] == max_attempt[rid] + 1  # incremented attempt number
    # the recovered request is now complete and no longer a recovery candidate
    assert main(["--phase", "judge", "--recover-failed", "--max-requests", "1", "--dry-run",
                 "--output", str(tmp_path)]) == 0
    assert len([e for e in _read_lines(ledger_path) if e["state"] == "completed"]) == len(
        [e for e in before if e["state"] == "completed"]) + 1


def test_repeated_recovery_selects_distinct_requests(tmp_path, fake_judge_backend):
    _synthetic_failed_requests(tmp_path, 3)
    ledger_path = tmp_path / "judging" / "judge_execution_ledger.jsonl"
    before = _read_lines(ledger_path)
    failed = {e["judge_request_id"] for e in before if e["state"] == "failed"}
    completed_before = {e["judge_request_id"] for e in before if e["state"] == "completed"}
    for _ in range(2):
        assert main(["--phase", "judge", "--recover-failed", "--max-requests", "1", "--execute",
                     "--confirm-paid-judge-calls", "--output", str(tmp_path)]) == 0
    completed_after = {e["judge_request_id"] for e in _read_lines(ledger_path) if e["state"] == "completed"}
    recovered = completed_after - completed_before
    assert len(recovered) == 2 and recovered.issubset(failed)


def test_interrupted_recovery_attempt_remains_fail_closed(tmp_path, fake_judge_backend):
    _synthetic_failed_requests(tmp_path, 1)
    ledger_path = tmp_path / "judging" / "judge_execution_ledger.jsonl"
    failed = next(e for e in _read_lines(ledger_path) if e["state"] == "failed")
    identity = {k: failed[k] for k in ("judge_request_id", "repeat_block_id", "judge_slot", "provider", "model",
                                       "source", "prompt_sha256", "clean_A_observation_id", "formal300_block_id",
                                       "clean_B_answer_sha256")}
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**identity, "state": "started", "attempt": 2}) + "\n")
    with pytest.raises(SystemExit, match="fail-closed"):
        main(["--phase", "judge", "--recover-failed", "--max-requests", "1", "--execute",
              "--confirm-paid-judge-calls", "--output", str(tmp_path)])
    assert fake_judge_backend == []


def test_recovery_failure_records_rich_diagnostics(tmp_path, monkeypatch):
    _synthetic_failed_requests(tmp_path, 1)
    import scripts.run_clean_repeat_stochasticity as mod

    def failing_backend(provider, model, prompt, source, max_output_tokens):
        exc = ValueError("malformed JSON: Unterminated string starting at: line 1 column 3425")
        exc.judge_diagnostics = {"provider": provider, "model": model, "error_kind": "malformed_json",
                                 "configured_max_output_tokens": max_output_tokens, "response_text_length": 3425,
                                 "appears_truncated": True}
        raise exc

    monkeypatch.setattr(mod, "_JUDGE_BACKEND", failing_backend)
    before = _read_lines(tmp_path / "judging" / "judge_execution_ledger.jsonl")
    assert main(["--phase", "judge", "--recover-failed", "--max-requests", "1", "--execute",
                 "--confirm-paid-judge-calls", "--output", str(tmp_path)]) == 0
    ledger = _read_lines(tmp_path / "judging" / "judge_execution_ledger.jsonl")
    new_failed = [e for e in ledger[len(before):] if e["state"] == "failed"]
    assert len(new_failed) == 1
    diagnostics = _read_lines(tmp_path / "judging" / "judge_diagnostics.jsonl")
    recorded = [d for d in diagnostics if d.get("state") == "failed"
                and d.get("judge_request_id") == new_failed[0]["judge_request_id"]]
    assert recorded and recorded[-1]["diagnostics"]["error_kind"] == "malformed_json"
    assert recorded[-1]["diagnostics"]["appears_truncated"] is True


def test_default_judge_backend_records_malformed_json_diagnostics(monkeypatch):
    import scripts.b4_calibration_execution as bce
    import scripts.run_clean_repeat_stochasticity as mod

    class FakeGemini:
        def __init__(self, model, max_output_tokens=1024, **kwargs):
            self.last_diagnostics = {"json_parse_ok": False, "raw_text_char_count": 3425,
                                     "raw_text_open_braces": 1, "raw_text_close_braces": 0,
                                     "output_tokens": 1024, "configured_max_output_tokens": max_output_tokens}
            self.last_metadata = {"finish_reason": "MAX_TOKENS", "output_tokens": 1024}

        def call(self, prompt, task):
            return '{"decisions": [{"criterion_id": "c1", "decision": "pass"'

    monkeypatch.setattr(bce, "GeminiJudgeAdapter", FakeGemini)
    prompt = json.dumps({"criteria": [{"criterion_id": "c1", "evaluation_type": "memory_presence",
                                       "criterion": "c", "expected_answer": "a"}]})
    with pytest.raises(ValueError) as info:
        default_judge_backend("gemini", "gemini-3-flash-preview", prompt, "memora", 1024)
    diag = info.value.judge_diagnostics
    assert diag["provider"] == "gemini" and diag["model"] == "gemini-3-flash-preview"
    assert diag["error_kind"] == "malformed_json"
    assert diag["configured_max_output_tokens"] == 1024
    assert diag["response_text_length"] is not None
    assert diag["appears_truncated"] is True
    assert "sk-" not in json.dumps(diag)


def test_default_judge_backend_records_empty_text_diagnostics(monkeypatch):
    import scripts.b4_calibration_execution as bce
    import scripts.run_clean_repeat_stochasticity as mod

    class FakeOpenAI:
        def __init__(self, model, max_output_tokens=1024, **kwargs):
            self.last_diagnostics = None

        def call(self, prompt, task):
            exc = ValueError("provider response contained no text")
            exc.diagnostics = {"status": "incomplete", "incomplete_reason": "max_output_tokens",
                               "output_tokens": 160, "configured_max_output_tokens": 160,
                               "content_block_types": [], "has_raw_text": False}
            self.last_diagnostics = exc.diagnostics
            raise exc

    monkeypatch.setattr(bce, "OpenAIJudgeAdapter", FakeOpenAI)
    with pytest.raises(ValueError) as info:
        default_judge_backend("openai", "gpt-5.6-sol", json.dumps({"task": "x"}), "longmemeval", 160)
    diag = info.value.judge_diagnostics
    assert diag["error_kind"] == "empty_text"
    assert diag["incomplete_reason"] == "max_output_tokens"
    assert diag["appears_truncated"] is True
    assert diag["response_text_length"] is None


def test_judge_output_budget_applies_frozen_floors():
    lme_openai = {"planned_output_tokens": 160, "provider": "openai", "judge_slot": "lme_primary_openai",
                  "source": "longmemeval", "phase": "primary"}
    lme_anthropic = {"planned_output_tokens": 160, "provider": "anthropic", "judge_slot": "lme_primary_anthropic",
                     "source": "longmemeval", "phase": "primary"}
    assert judge_output_budget(lme_openai, "planned") == 160
    assert judge_output_budget(lme_openai, "formal300") == 512
    assert judge_output_budget(lme_anthropic, "planned") == 160
    assert judge_output_budget(lme_anthropic, "formal300") == 2048
    memora = {"planned_output_tokens": planned_output_tokens(
        "memora", json.dumps({"criteria": [{"criterion_id": str(i)} for i in range(56)]})),
        "provider": "gemini", "judge_slot": "memora_primary_gemini", "source": "memora", "phase": "primary"}
    assert memora["planned_output_tokens"] == 40 + 36 * 56
    assert judge_output_budget(memora, "formal300") == 40 + 36 * 56


# ---------------------------------------------------------------------------
# Gemini Memora output-budget / truncation audit
# ---------------------------------------------------------------------------

def test_output_budget_override_is_explicit_and_transport_only():
    row = {"planned_output_tokens": 2056, "provider": "gemini", "judge_slot": "memora_primary_gemini",
           "source": "memora", "phase": "primary", "criterion_count": 56}
    default = output_budget_record(row, "formal300", "none")
    assert default["planned_output_tokens"] == 2056
    assert default["effective_output_tokens"] == 2056
    assert default["output_budget_policy"] == "formal300_floors"
    override = output_budget_record(row, "formal300", "gemini_memora_transport")
    assert override["planned_output_tokens"] == 2056  # frozen formal-300 budget preserved as metadata
    assert override["effective_output_tokens"] == 4736  # min(6144, max(1024, 256 + 80*56))
    assert override["output_budget_policy"] == "formal300_recovery_256_80n"


def test_output_budget_override_only_touches_gemini_memora():
    openai_row = {"planned_output_tokens": 512, "provider": "openai", "judge_slot": "lme_primary_openai",
                  "source": "longmemeval", "phase": "primary", "criterion_count": None}
    assert output_budget_record(openai_row, "formal300", "gemini_memora_transport")["effective_output_tokens"] == 512
    anthropic_row = {"planned_output_tokens": 160, "provider": "anthropic", "judge_slot": "lme_primary_anthropic",
                     "source": "longmemeval", "phase": "primary", "criterion_count": None}
    assert output_budget_record(anthropic_row, "formal300",
                                "gemini_memora_transport")["effective_output_tokens"] == 2048


def test_gemini_memora_transport_budget_matches_frozen_policy():
    assert gemini_memora_transport_budget(56) == 4736
    assert gemini_memora_transport_budget(59) == 4976
    assert gemini_memora_transport_budget(86) == 6144  # capped at the repository ceiling
    assert gemini_memora_transport_budget(2) == 1024   # floor


def test_gemini_max_tokens_truncation_diagnostics(monkeypatch):
    import scripts.b4_calibration_execution as bce
    import scripts.run_clean_repeat_stochasticity as mod

    class FakeGemini:
        def __init__(self, model, max_output_tokens=1024, **kwargs):
            self.last_diagnostics = None
            self.last_metadata = {"finish_reason": "FinishReason.MAX_TOKENS",
                                  "configured_max_output_tokens": max_output_tokens}

        def call(self, prompt, task):
            return '{"decisions": [{"criterion_id": "c1", "decision": "pass"'

    monkeypatch.setattr(bce, "GeminiJudgeAdapter", FakeGemini)
    prompt = json.dumps({"criteria": [{"criterion_id": "c1", "evaluation_type": "memory_presence",
                                       "criterion": "c", "expected_answer": "a"}]})
    with pytest.raises(ValueError) as info:
        default_judge_backend("gemini", "gemini-3-flash-preview", prompt, "memora", 2056)
    diag = info.value.judge_diagnostics
    assert diag["error_kind"] == "malformed_json"
    assert diag["finish_reason"] == "FinishReason.MAX_TOKENS"
    assert diag["configured_max_output_tokens"] == 2056
    assert diag["appears_truncated"] is True


def _budget_capturing_backend(monkeypatch):
    import scripts.run_clean_repeat_stochasticity as mod
    calls: list[dict] = []

    def backend(provider, model, prompt, source, max_output_tokens):
        calls.append({"provider": provider, "source": source, "max_output_tokens": max_output_tokens})
        if source == "longmemeval":
            parsed = {"decision": "correct", "confidence": "high", "short_reason": "ok", "exploratory_binary": True}
        else:
            parsed = {"decisions": [], "native_fama": 1.0, "exploratory_binary": True}
        return {"raw": json.dumps(parsed), "parsed": parsed, "diagnostics": {"provider": provider}}

    monkeypatch.setattr(mod, "_JUDGE_BACKEND", backend)
    return calls


def test_recovery_default_and_override_budgets(tmp_path, monkeypatch):
    row = _synthetic_failed_requests(tmp_path, 1)[0]
    calls = _budget_capturing_backend(monkeypatch)
    # default (no override): the frozen formal-300 plan budget is used
    assert main(["--phase", "judge", "--recover-failed", "--judge-provider", "gemini", "--max-requests", "1",
                 "--execute", "--confirm-paid-judge-calls", "--output", str(tmp_path)]) == 0
    assert calls[-1]["provider"] == "gemini"
    assert calls[-1]["max_output_tokens"] == row["planned_output_tokens"]


def test_recovery_with_override_records_effective_budget(tmp_path, monkeypatch):
    row = _synthetic_failed_requests(tmp_path, 1)[0]
    expected = gemini_memora_transport_budget(row["criterion_count"])
    calls = _budget_capturing_backend(monkeypatch)
    assert main(["--phase", "judge", "--recover-failed", "--judge-provider", "gemini",
                 "--output-budget-override", "gemini_memora_transport", "--max-requests", "1",
                 "--execute", "--confirm-paid-judge-calls", "--output", str(tmp_path)]) == 0
    # the batch now requests the frozen transport budget rather than the plan budget
    assert calls[-1]["provider"] == "gemini" and calls[-1]["max_output_tokens"] == expected
    diagnostics = _read_lines(tmp_path / "judging" / "judge_diagnostics.jsonl")
    recorded = [d for d in diagnostics if d.get("output_budget_policy") == "formal300_recovery_256_80n"]
    assert recorded
    last = recorded[-1]
    assert last["planned_output_tokens"] == row["planned_output_tokens"]  # frozen budget preserved
    assert last["effective_output_tokens"] == expected                    # explicit override applied
    assert last["criterion_count"] == row["criterion_count"]


def test_failure_ledger_is_preserved_across_failed_recovery(tmp_path, monkeypatch):
    _synthetic_failed_requests(tmp_path, 1)
    ledger_path = tmp_path / "judging" / "judge_execution_ledger.jsonl"
    before = _read_lines(ledger_path)
    failed_before = [e for e in before if e["state"] == "failed"]
    import scripts.run_clean_repeat_stochasticity as mod

    def failing_backend(provider, model, prompt, source, max_output_tokens):
        exc = ValueError("malformed JSON: Unterminated string starting at: line 1 column 3425")
        exc.judge_diagnostics = {"provider": provider, "model": model, "error_kind": "malformed_json",
                                 "finish_reason": "FinishReason.MAX_TOKENS",
                                 "configured_max_output_tokens": max_output_tokens, "appears_truncated": True}
        raise exc

    monkeypatch.setattr(mod, "_JUDGE_BACKEND", failing_backend)
    assert main(["--phase", "judge", "--recover-failed", "--judge-provider", "gemini", "--max-requests", "1",
                 "--execute", "--confirm-paid-judge-calls", "--output", str(tmp_path)]) == 0
    after = _read_lines(ledger_path)
    assert after[:len(before)] == before  # append-only: original failure records preserved
    new_failed = [e for e in after[len(before):] if e["state"] == "failed"]
    assert len(new_failed) == 1
    assert new_failed[0]["attempt"] == max(e.get("attempt", 1) for e in before if
                                           e["judge_request_id"] == new_failed[0]["judge_request_id"]) + 1
    # the request remains a recoverable failure, never marked complete
    results_path = tmp_path / "judging" / "judge_results.jsonl"
    result_rows = _read_lines(results_path) if results_path.exists() else []
    state = reconcile_judge_state(set(), after, result_rows)
    assert state["requests"][new_failed[0]["judge_request_id"]]["status"] == "failed"
    assert state["requests"][new_failed[0]["judge_request_id"]]["recoverable"] is True
