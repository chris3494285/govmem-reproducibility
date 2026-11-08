from __future__ import annotations

import json
from pathlib import Path

from scripts.build_governance_fault_rq1_restrictive_manifest import build_from_candidates
from scripts.score_governance_fault_rq1 import contains_exact_value, normalize_text, score_files, score_rows
from tests.test_governance_fault_rq1_restrictive_manifest import _candidate


def _manifest_rows():
    return build_from_candidates([_candidate("memora", i) for i in range(15)] + [_candidate("memconflict", i) for i in range(15)])


def test_format_only_target_matching_accepts_native_underscores_and_prose():
    target = "Spending_Less_Time_Socializing"
    assert contains_exact_value("spending less time socializing", target) is True
    assert contains_exact_value(target, target) is True
    assert contains_exact_value("spending more time socializing", target) is False
    assert normalize_text("A-B__C") == "a b c"


def test_memconflict_dict_target_uses_native_leaf_and_exact_gpt_smoke_wording():
    target = {"Current_State": "Spending_Less_Time_Socializing"}
    answer = (
        "Yes. The user's social status changed from keeping to oneself to "
        "spending less time socializing."
    )
    assert contains_exact_value(answer, target) is True


def test_rq1b_social_status_pair_preserves_paired_false_denial_logic():
    rows = [
        json.loads(line)
        for line in Path("data/governance_fault_rq1_restrictive_frozen.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Select the temporal Social_Status pair by structure rather than by a
    # source-derived example id: the public test source never embeds benchmark ids.
    social = [
        row for row in rows
        if row.get("source_dataset") == "memconflict"
        and row.get("dimension") == "temporal"
        and row.get("attribute") == "Social_Status"
    ]
    example_id = social[0]["example_id"]
    manifest = [row for row in social if row["example_id"] == example_id]
    assert len(manifest) == 2
    clean, fault = sorted(manifest, key=lambda row: row["condition_type"] == "fault")
    answer = (
        "Yes. The user's social status changed from keeping to oneself to "
        "spending less time socializing."
    )
    inference = [
        dict(clean, model_answer=answer, status="success", actual_admission="allow"),
        dict(fault, model_answer="It cannot be determined from the available context.", status="success", actual_admission="deny"),
    ]
    scored = score_rows(inference, {row["run_id"]: row for row in manifest})
    assert scored[0]["target_utility"] is True
    assert scored[0]["mentions_required_target"] is True
    assert scored[1]["target_utility"] is False
    assert scored[1]["downstream_false_denial"] is True


def test_target_level_memora_false_denial_and_utility_delta():
    manifest = _manifest_rows()
    group = [row for row in manifest if row["example_id"] == "memora::0"]
    clean, fault = sorted(group, key=lambda row: row["condition_type"] == "fault")
    inference = [
        dict(clean, model_answer="The active item is active item 0.", status="success", actual_admission="allow"),
        dict(fault, model_answer="It cannot be determined from the available context.", status="success", actual_admission="deny"),
    ]
    scored = score_rows(inference, {row["run_id"]: row for row in manifest})
    assert scored[0]["mentions_required_target"] is True
    assert scored[0]["target_utility"] is True
    assert scored[1]["control_plane_false_rejection"] is True
    assert scored[1]["downstream_false_denial"] is True
    assert scored[1]["target_utility"] is False


def test_target_level_memconflict_false_denial_and_utility_delta():
    manifest = _manifest_rows()
    group = [row for row in manifest if row["example_id"] == "conflict::0"]
    clean, fault = sorted(group, key=lambda row: row["condition_type"] == "fault")
    inference = [
        dict(clean, model_answer="The current value is Melbourne, Australia.", status="success", actual_admission="allow"),
        dict(fault, model_answer="It cannot be determined.", status="success", actual_admission="deny"),
    ]
    scored = score_rows(inference, {row["run_id"]: row for row in manifest})
    assert scored[1]["downstream_false_denial"] is True
    assert scored[0]["target_utility"] is True
    assert scored[1]["target_utility"] is False


def test_clean_baseline_without_target_makes_paired_false_denial_unavailable():
    manifest = _manifest_rows()
    group = [row for row in manifest if row["example_id"] == "memora::0"]
    clean, fault = sorted(group, key=lambda row: row["condition_type"] == "fault")
    inference = [
        dict(clean, model_answer="No usable information.", status="success", actual_admission="allow"),
        dict(fault, model_answer="No usable information.", status="success", actual_admission="deny"),
    ]
    scored = score_rows(inference, {row["run_id"]: row for row in manifest})
    assert scored[1]["downstream_false_denial"] is None


def test_rq1b_detection_uses_manifest_and_clean_is_excluded_from_false_denial_rate(tmp_path):
    manifest = _manifest_rows()
    group = [row for row in manifest if row["example_id"] == "memora::0"]
    clean, fault = sorted(group, key=lambda row: row["condition_type"] == "fault")
    inference = []
    for row, answer, admission in (
        (clean, "The active item is active item 0.", "allow"),
        (fault, "It cannot be determined.", "deny"),
    ):
        item = dict(row)
        item.pop("manifest_version", None)
        item.update({"model_answer": answer, "status": "success", "actual_admission": admission})
        inference.append(item)
    manifest_map = {row["run_id"]: row for row in manifest}
    scored = score_rows(inference, manifest_map)
    assert scored[0]["downstream_false_denial"] is None
    assert scored[1]["downstream_false_denial"] is True
    assert all("_scoring_manifest_version" not in row for row in scored)

    input_path = tmp_path / "inference.jsonl"
    output_path = tmp_path / "scored.jsonl"
    summary_path = tmp_path / "summary.json"
    input_path.write_text("".join(json.dumps(row) + "\n" for row in inference))
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in manifest))
    summary = score_files(manifest_path, input_path, output_path, summary_path)
    assert summary["overall"]["scored_downstream_false_denial_n"] == 1
    assert summary["overall"]["downstream_false_denial_rate"] == 1.0
