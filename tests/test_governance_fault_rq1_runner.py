import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from eval.reconstruction_provider import ProviderResponse, ProviderTelemetry, ProviderTransportError
from govmem.governance.enforcement import decide_admission
from scripts.run_governance_fault_rq1 import (
    NEUTRAL_CONTEXT_INSTRUCTION,
    construct_context,
    load_manifest,
    run_manifest,
    select_base_rows,
    validate_manifest_rows,
)


def _condition(dimension, state, *, memory=None, gold=None):
    return {
        "dimension": dimension,
        "observed_governance_state": state,
        "gold_governance_state": gold if gold is not None else state,
        "memory": memory or [{"interaction_id": "m1", "content": "SECRET_TARGET"}],
    }


@pytest.mark.parametrize(
    "dimension,state,expected,reason",
    [
        ("authorization", {"authorization": {"decision": "allow"}, "authorization_assertions": []}, "allow", "authorization_allow"),
        ("authorization", {"authorization": {"decision": "deny"}, "authorization_assertions": []}, "deny", "authorization_deny"),
        ("authorization", {"authorization": None, "authorization_assertions": []}, "allow", "authorization_missing_permissive_fallback"),
        ("authorization", {"authorization": None, "authorization_assertions": [{"decision": "allow"}, {"decision": "deny"}]}, "allow", "authorization_conflict_permissive_fallback"),
        ("deletion", {"deletion": {"status": "active"}}, "allow", "deletion_active"),
        ("deletion", {"deletion": {"status": "deleted"}}, "deny", "deletion_deleted"),
        ("deletion", {"deletion": None}, "allow", "deletion_missing_permissive_fallback"),
        ("temporal", {"temporal": {"status": "valid"}, "temporal_assertions": []}, "allow", "temporal_valid"),
        ("temporal", {"temporal": {"status": "expired"}, "temporal_assertions": []}, "deny", "temporal_expired"),
        ("temporal", {"temporal": None, "temporal_assertions": []}, "allow", "temporal_missing_permissive_fallback"),
        ("temporal", {"temporal": None, "temporal_assertions": [{"status": "valid"}, {"status": "expired"}]}, "allow", "temporal_conflict_permissive_fallback"),
    ],
)
def test_permissive_policy(dimension, state, expected, reason):
    result = decide_admission(_condition(dimension, state))
    assert result.actual_admission == expected
    assert result.enforcement_reason == reason
    assert result.fallback_used is ("fallback" in reason)


def test_policy_uses_observed_not_gold_and_denied_context_hides_memory():
    row = _condition("authorization", {"authorization": {"decision": "allow"}, "authorization_assertions": []}, gold={"authorization": {"decision": "deny"}, "authorization_assertions": []})
    result = decide_admission(row)
    assert result.actual_admission == "allow"
    denied = _condition("deletion", {"deletion": {"status": "deleted"}})
    admission = decide_admission(denied)
    prompt, selected, rejected = construct_context(denied, admission)
    assert "SECRET_TARGET" not in prompt
    assert NEUTRAL_CONTEXT_INSTRUCTION in prompt
    assert selected == []
    assert rejected == ["m1"]


def test_neutral_instruction_is_identical_in_allow_and_deny_prompts_without_gold_fields():
    allow = _condition("authorization", {"authorization": {"decision": "allow"}, "authorization_assertions": []})
    deny = _condition("authorization", {"authorization": {"decision": "deny"}, "authorization_assertions": []})
    allow_prompt, _, _ = construct_context(allow, decide_admission(allow))
    deny_prompt, _, _ = construct_context(deny, decide_admission(deny))
    assert NEUTRAL_CONTEXT_INSTRUCTION in allow_prompt
    assert NEUTRAL_CONTEXT_INSTRUCTION in deny_prompt
    assert allow_prompt.count(NEUTRAL_CONTEXT_INSTRUCTION) == 1
    assert deny_prompt.count(NEUTRAL_CONTEXT_INSTRUCTION) == 1
    for forbidden in ("gold_governance_state", "expected_action", "leak_targets", "authorization", "deletion", "temporal"):
        assert forbidden not in allow_prompt.lower()
        assert forbidden not in deny_prompt.lower()


def test_frozen_manifest_limit_is_base_level_and_deterministic():
    rows = load_manifest("data/governance_fault_rq1_pilot_frozen.jsonl")
    selected_a = select_base_rows(rows, 3)
    selected_b = select_base_rows(rows, 3)
    assert len(selected_a) == 9
    assert [row["run_id"] for row in selected_a] == [row["run_id"] for row in selected_b]
    assert len({row["example_id"] for row in selected_a}) == 3


def test_rq1a_requires_three_condition_groups():
    rows = load_manifest("data/governance_fault_rq1_pilot_frozen.jsonl")
    validate_manifest_rows(rows, require_full=True)
    first_id = sorted({row["example_id"] for row in rows})[0]
    # Remove exactly one row from the first group without changing the
    # historical manifest on disk.
    shortened = [row for row in rows if row["run_id"] != next(row["run_id"] for row in rows if row["example_id"] == first_id and row["condition_type"] == "fault")]
    with pytest.raises(ValueError, match="exactly 3 conditions"):
        validate_manifest_rows(shortened, require_full=False)


def test_rq1b_requires_two_rows_one_clean_and_one_restrictive_fault():
    rows = load_manifest("data/governance_fault_rq1_restrictive_frozen.jsonl")
    validate_manifest_rows(rows, require_full=True)

    first_id = sorted({row["example_id"] for row in rows})[0]
    missing_fault = [row for row in rows if not (row["example_id"] == first_id and row["condition_type"] == "fault")]
    with pytest.raises(ValueError, match="exactly 2 conditions"):
        validate_manifest_rows(missing_fault, require_full=False)

    extra = deepcopy(rows[0])
    extra["run_id"] = "extra-rq1b-row"
    extra["condition_type"] = "fault"
    extra["fault_spec"] = {"dimension": rows[0]["dimension"], "operator": "flipped", "direction": "restrictive"}
    with pytest.raises(ValueError, match="exactly 2 conditions"):
        validate_manifest_rows(rows + [extra], require_full=False)


def test_rq1b_limit_is_base_level_and_selects_four_rows(tmp_path):
    output = tmp_path / "rq1b-preflight.jsonl"
    results = run_manifest(
        manifest="data/governance_fault_rq1_restrictive_frozen.jsonl",
        output=output,
        limit=2,
        dry_run=True,
        overwrite=True,
    )
    assert len(results) == 4
    assert len({row["example_id"] for row in results}) == 2
    assert all(row["status"] == "dry_run" for row in results)


class FakeDownstream:
    def __init__(self, model, *, fail=False):
        self.model = model
        self.fail = fail
        self.prompts = []

    def request_downstream(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            raise ProviderTransportError("mock failure", provider="deepseek", model=self.model, details={"api_latency_ms": 12.5, "request_id": "req-fail"})
        return ProviderResponse({"answer": "mock answer"}, ProviderTelemetry("deepseek", self.model, 10, 4, 14, 2.5, "req-1", "stop"))


def test_dry_run_makes_zero_provider_calls_and_writes_nine_rows(tmp_path):
    called = []
    out = tmp_path / "dry.jsonl"
    results = run_manifest(manifest="data/governance_fault_rq1_pilot_frozen.jsonl", output=out, limit=3, dry_run=True, provider_factory=lambda model: called.append(model))
    assert len(results) == 9
    assert called == []
    assert all(row["status"] == "dry_run" for row in results)


def test_provider_telemetry_and_failure_are_persisted(tmp_path):
    fake = FakeDownstream("deepseek-v4-pro")
    out = tmp_path / "results.jsonl"
    results = run_manifest(manifest="data/governance_fault_rq1_pilot_frozen.jsonl", output=out, limit=1, provider_factory=lambda model: fake)
    assert len(results) == 3
    assert results[0]["provider"] == "deepseek"
    assert results[0]["input_tokens"] == 10
    assert results[0]["total_tokens"] == 14
    assert results[0]["latency_ms"] == 2.5
    assert results[0]["selection_reason"]
    assert set(json.loads(out.read_text().splitlines()[0])) >= {
        "run_id", "example_id", "actual_admission", "constructed_context", "model_answer", "status", "error",
    }
    failed = FakeDownstream("deepseek-v4-pro", fail=True)
    failed_out = tmp_path / "failed.jsonl"
    failed_results = run_manifest(manifest="data/governance_fault_rq1_pilot_frozen.jsonl", output=failed_out, limit=1, provider_factory=lambda model: failed)
    assert len(failed_results) == 3
    assert all(row["status"] == "provider_error" for row in failed_results)
    assert failed_results[0]["latency_ms"] == 12.5


def test_identical_prompts_reuse_only_within_one_base_trio(tmp_path):
    first_group = _frozen_group("authorization", "flipped")
    first_id = first_group["clean"]["example_id"]
    rows = load_manifest("data/governance_fault_rq1_pilot_frozen.jsonl")
    # Use the first deterministic base plus a second base; the two fault rows
    # in the first base have identical admitted content, while base IDs differ.
    ids = sorted({row["example_id"] for row in rows})
    selected_ids = {first_id, ids[1]}
    subset = [row for row in rows if row["example_id"] in selected_ids]
    manifest = tmp_path / "subset.jsonl"
    manifest.write_text("\n".join(json.dumps(row, sort_keys=True) for row in subset) + "\n")
    fake = FakeDownstream("deepseek-v4-pro")
    output = tmp_path / "dedup.jsonl"
    results = run_manifest(manifest=manifest, output=output, limit=2, provider_factory=lambda model: fake, temperature=0)
    assert len(results) == 6
    expected_calls = sum(
        len({row["prompt_sha256"] for row in results if row["example_id"] == example_id})
        for example_id in selected_ids
    )
    assert len(fake.prompts) == expected_calls
    reused = [row for row in results if row["response_reused"]]
    assert reused
    for row in reused:
        source = next(item for item in results if item["run_id"] == row["reused_from_run_id"])
        assert row["model_answer"] == source["model_answer"]
        assert row["input_tokens"] == source["input_tokens"]
        assert row["request_id"] == source["request_id"]
        assert row["prompt_sha256"] == source["prompt_sha256"]
    assert all(row["temperature"] == 0 for row in results)


def test_different_temperature_is_part_of_generation_identity(tmp_path):
    rows = load_manifest("data/governance_fault_rq1_pilot_frozen.jsonl")
    first_id = sorted({row["example_id"] for row in rows})[0]
    manifest = tmp_path / "one.jsonl"
    manifest.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows if row["example_id"] == first_id) + "\n")
    fake = FakeDownstream("deepseek-v4-pro")
    results = run_manifest(manifest=manifest, output=tmp_path / "temp.jsonl", limit=1, provider_factory=lambda model: fake, temperature=0.2)
    assert all(row["temperature"] == 0.2 for row in results)


def _frozen_group(dimension, operator=None):
    rows = load_manifest("data/governance_fault_rq1_pilot_frozen.jsonl")
    groups = {}
    for row in rows:
        if row["dimension"] == dimension:
            groups.setdefault(row["example_id"], []).append(row)
    for group in groups.values():
        if operator is None or any((row.get("fault_spec") or {}).get("operator") == operator for row in group):
            return {row["condition_type"] if row["condition_type"] == "clean" else (row["fault_spec"] or {}).get("operator"): row for row in group}
    raise AssertionError(f"No frozen {dimension}/{operator} group")


def test_frozen_temporal_clean_stale_and_conflict_use_distinct_memory_projections():
    group = _frozen_group("temporal", "conflicting")
    clean_admission = decide_admission(group["clean"])
    stale_admission = decide_admission(group["stale"])
    conflict_admission = decide_admission(group["conflicting"])
    clean_prompt, clean_selected, clean_rejected = construct_context(group["clean"], clean_admission)
    stale_prompt, stale_selected, stale_rejected = construct_context(group["stale"], stale_admission)
    conflict_prompt, conflict_selected, conflict_rejected = construct_context(group["conflicting"], conflict_admission)
    metadata = group["clean"]["source_metadata"]
    current = json.dumps(metadata["current_value"], sort_keys=True)
    previous = json.dumps(metadata["previous_value"], sort_keys=True)
    assert current in clean_prompt
    assert previous not in clean_prompt
    assert clean_admission.actual_admission == "allow"
    assert clean_selected and clean_rejected
    assert previous in stale_prompt
    assert current not in stale_prompt
    assert stale_admission.actual_admission == "allow"
    assert stale_admission.enforcement_reason == "temporal_stale_predecessor_admitted"
    assert stale_selected and stale_rejected
    assert current in conflict_prompt and previous in conflict_prompt
    assert conflict_admission.actual_admission == "allow"
    assert conflict_selected and not conflict_rejected
    assert "gold" not in conflict_prompt.lower()
    assert "winner" not in conflict_prompt.lower()
    assert clean_prompt != stale_prompt
    assert clean_prompt != conflict_prompt


def test_rq1b_memconflict_restrictive_context_keeps_comparison_evidence_only():
    rows = load_manifest("data/governance_fault_rq1_restrictive_frozen.jsonl")
    group_rows = [row for row in rows if row["source_dataset"] == "memconflict"]
    example_id = sorted({row["example_id"] for row in group_rows})[0]
    group = {row["condition_type"]: row for row in group_rows if row["example_id"] == example_id}

    clean_admission = decide_admission(group["clean"])
    fault_admission = decide_admission(group["fault"])
    clean_prompt, clean_selected, clean_rejected = construct_context(group["clean"], clean_admission)
    fault_prompt, fault_selected, fault_rejected = construct_context(group["fault"], fault_admission)
    metadata = group["clean"]["source_metadata"]
    current = json.dumps(metadata["current_value"], ensure_ascii=False, sort_keys=True)
    previous = json.dumps(metadata["previous_value"], ensure_ascii=False, sort_keys=True)

    assert current in clean_prompt and previous in clean_prompt
    assert '"role": "current"' in clean_prompt
    assert '"role": "previous"' in clean_prompt
    assert clean_selected == [
        f"{metadata['attribute']}@{metadata['previous_session']['session_id']}",
        f"{metadata['attribute']}@{metadata['current_session']['session_id']}",
    ]
    assert clean_rejected == []

    assert fault_admission.actual_admission == "deny"
    assert previous in fault_prompt and current not in fault_prompt
    assert '"role": "previous"' in fault_prompt
    assert '"role": "current"' not in fault_prompt
    assert fault_selected == [f"{metadata['attribute']}@{metadata['previous_session']['session_id']}"]
    assert fault_rejected == [f"{metadata['attribute']}@{metadata['current_session']['session_id']}"]
    assert group["clean"]["task"] in clean_prompt and group["fault"]["task"] in fault_prompt
    assert NEUTRAL_CONTEXT_INSTRUCTION in clean_prompt
    assert NEUTRAL_CONTEXT_INSTRUCTION in fault_prompt
    for forbidden in ("gold", "fault", "restrictive", "expired", "evaluator"):
        assert forbidden not in clean_prompt.lower()
        assert forbidden not in fault_prompt.lower()
    assert clean_prompt != fault_prompt


def test_denied_rows_never_select_memory():
    rows = load_manifest("data/governance_fault_rq1_pilot_frozen.jsonl")
    for row in rows:
        admission = decide_admission(row)
        _, selected, _ = construct_context(row, admission)
        if admission.actual_admission == "deny":
            assert selected == []


def test_frozen_authorization_and_deletion_faults_change_selection():
    auth = _frozen_group("authorization", "flipped")
    clean_auth = construct_context(auth["clean"], decide_admission(auth["clean"]))
    flipped_auth = construct_context(auth["flipped"], decide_admission(auth["flipped"]))
    assert clean_auth[1] == [] and clean_auth[2]
    assert flipped_auth[1] and flipped_auth[2] == []
    deletion = _frozen_group("deletion", "stale")
    clean_del = construct_context(deletion["clean"], decide_admission(deletion["clean"]))
    stale_del = construct_context(deletion["stale"], decide_admission(deletion["stale"]))
    assert clean_del[1] == []
    assert stale_del[1]
    assert "Prepare lecture materials" in stale_del[0] or stale_del[1]
