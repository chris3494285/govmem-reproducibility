from eval.rq1_metrics import evaluate_records


def _record(
    query_id,
    dataset,
    task_type,
    dimensions,
    states=None,
    *,
    skipped=False,
    status="success",
    validation="valid",
    oracle_available=False,
    oracle_fields=None,
    oracles=None,
    gold=None,
):
    return {
        "query_id": query_id,
        "dataset": dataset,
        "task_type": task_type,
        "applicable_governance_dimensions": dimensions,
        "skipped": skipped,
        "reconstruction_status": status,
        "validation_status": validation,
        "reconstructed_states": states or [],
        "evaluator_only": {
            "model_visible_interaction_ids": ["i1", "i2"],
            "oracle_available": oracle_available,
            "oracle_fields_supported": oracle_fields or [],
            "oracle_governance": oracles or [],
            "evaluator_metadata": {"gold": gold or {}},
        },
    }


def test_rq1_separates_gate_privacy_and_safety_empty_reconstruction():
    summary = evaluate_records([
        _record("privacy", "gatemem", "privacy", ["authorization_scope"], states=[{
            "interaction_id": "i1", "authorized_scopes": [], "missing_fields": ["authorized_scopes"],
        }], gold={"expected_action": "refuse"}),
        _record("safety", "gatemem", "safety", ["lifecycle_deletion"], gold={"expected_action": "refuse"}),
    ])
    assert summary["structural"]["n_examples"] == 2
    assert summary["by_governance_dimension"]["gatemem:privacy:authorization_scope"]["empty_reconstruction_rate"] == 0
    assert summary["by_governance_dimension"]["gatemem:safety:lifecycle_deletion"]["empty_reconstruction_rate"] == 1
    assert summary["native_supported_state_agreement"]["authorized_scopes"]["n"] == 0
    assert summary["native_supported_state_agreement"]["is_deleted"]["n"] == 0


def test_memora_lifecycle_missing_field_and_conflict_dimensions():
    summary = evaluate_records([
        _record("memora-1", "memora", "forgetting", ["lifecycle_deletion"], states=[{
            "interaction_id": "i1", "is_deleted": None, "missing_fields": ["is_deleted"],
        }], gold={"evaluation": {"action": "forget"}}),
        _record("dynamic", "memconflict", "dynamic", ["temporal_validity"], states=[{
            "interaction_id": "i1", "valid_from": "2024-01-01", "valid_until": None,
            "missing_fields": ["valid_until"],
        }], gold={"gold_answer": "Seattle"}),
        _record("conditional", "memconflict", "conditional", ["contextual_validity"], states=[{
            "interaction_id": "i1", "metadata": {"contextually_valid": True}, "missing_fields": [],
        }], gold={"gold_answer": "yes"}),
        _record("static", "memconflict", "static", [], skipped=True, status="skipped", validation="not_applicable"),
    ])
    assert summary["structural"]["n_examples"] == 3
    assert "memconflict:static" not in " ".join(summary["by_task_type"])
    labels = {row["query_id"]: row["reconstruction_error_labels"] for row in summary["per_example"]}
    assert "missing_deletion_state" in labels["memora-1"]
    assert "missing_temporal_state" not in labels["dynamic"]
    assert labels["static"] == []
    assert all(row["downstream_prediction_status"] == "not_run" for row in summary["per_example"])


def test_provider_error_is_not_valid_empty_and_downstream_gold_is_pending():
    summary = evaluate_records([_record(
        "error", "gatemem", "privacy", ["authorization_scope"],
        status="error", validation="transport_error", gold={"expected_action": "answer"},
    )])
    row = summary["per_example"][0]
    assert row["reconstruction_error_labels"] == ["invalid_prediction", "provider_error"]
    assert row["downstream_gold_available"] is True
    assert row["downstream_correct"] is None
    assert summary["structural"]["valid_prediction_rate"] == 0
    assert summary["structural"]["empty_reconstruction_rate"] == 0
    assert summary["structural"]["provider_execution_failure_count"] == 1
    assert summary["structural"]["model_reconstruction_failure_count"] == 0


def test_supported_oracle_fields_alone_enable_field_agreement():
    summary = evaluate_records([_record(
        "oracle", "synthetic", "lifecycle", ["lifecycle_deletion"], states=[{
            "interaction_id": "i1", "is_deleted": True, "missing_fields": [],
        }], oracle_available=True, oracle_fields=["is_deleted"], oracles=[{
            "interaction_id": "i1", "is_deleted": True,
        }],
    )])
    metric = summary["native_supported_state_agreement"]["is_deleted"]
    assert metric["n"] == 1
    assert metric["accuracy"] == 1
    assert summary["native_supported_state_agreement"]["authorized_scopes"]["n"] == 0


def test_relevant_completeness_excludes_irrelevant_schema_fields():
    summary = evaluate_records([
        _record("dynamic", "memconflict", "dynamic", ["temporal_validity"], states=[{
            "interaction_id": "i1",
            "valid_from": "2024-01-01",
            "valid_until": "2024-02-01",
            "missing_fields": [],
        }]),
        _record("privacy", "gatemem", "privacy", ["authorization_scope"], states=[{
            "interaction_id": "i1",
            "authorized_scopes": ["employee"],
            "missing_fields": ["is_deleted", "valid_from", "valid_until"],
        }]),
        _record("context", "memconflict", "conditional", ["contextual_validity"], states=[{
            "interaction_id": "i1",
            "metadata": {"contextually_valid": True},
            "missing_fields": [],
        }]),
    ])
    dynamic = summary["by_task_type"]["memconflict:dynamic"]
    privacy = summary["by_task_type"]["gatemem:privacy"]
    context = summary["by_task_type"]["memconflict:conditional"]
    assert dynamic["relevant_field_missing_rate"] == 0
    assert privacy["relevant_field_missing_rate"] == 0
    assert context["relevant_field_missing_rates"]["contextually_valid"] == 0
    assert privacy["schema_missing_field_rate"] > 0
    assert "missing_governance_field" not in next(
        row["reconstruction_error_labels"] for row in summary["per_example"] if row["query_id"] == "privacy"
    )


def test_grounding_reports_correct_outside_missing_and_unavailable_cases():
    grounded = _record("grounded", "gatemem", "privacy", ["authorization_scope"], states=[{
        "interaction_id": "i1", "authorized_scopes": ["employee"], "missing_fields": [],
    }])
    outside = _record("outside", "gatemem", "privacy", ["authorization_scope"], states=[{
        "interaction_id": "not-visible", "authorized_scopes": ["employee"], "missing_fields": [],
    }])
    missing = _record("missing", "gatemem", "privacy", ["authorization_scope"], states=[{
        "authorized_scopes": ["employee"], "missing_fields": [],
    }])
    unavailable = _record("unavailable", "gatemem", "privacy", ["authorization_scope"], states=[{
        "interaction_id": "i1", "authorized_scopes": ["employee"], "missing_fields": [],
    }])
    unavailable["evaluator_only"].pop("model_visible_interaction_ids")
    summary = evaluate_records([grounded, outside, missing, unavailable])
    metrics = summary["structural"]
    assert metrics["grounding_denominator"] == 2
    assert metrics["grounding_coverage"] == 1 / 2
    assert metrics["grounding_missing_interaction_id"] == 1
    assert metrics["grounding_outside_accepted_set"] == 1
    labels = {row["query_id"]: row["reconstruction_error_labels"] for row in summary["per_example"]}
    assert "wrong_or_unsupported_grounding" not in labels["grounded"]
    assert "wrong_or_unsupported_grounding" in labels["outside"]
    assert "missing_interaction_id" in labels["missing"]
    assert "wrong_or_unsupported_grounding" not in labels["unavailable"]


def test_temporal_null_endpoint_is_not_missing_without_explicit_marker():
    summary = evaluate_records([_record(
        "unbounded", "memconflict", "dynamic", ["temporal_validity"], states=[{
            "interaction_id": "i1", "valid_from": "2024-01-01", "valid_until": None, "missing_fields": [],
        }]
    )])
    metrics = summary["structural"]
    assert metrics["relevant_field_denominator"] == 1
    assert metrics["relevant_field_missing_rate"] == 0


def test_temporal_missing_all_usable_information_is_labeled_missing():
    summary = evaluate_records([_record(
        "no-temporal", "memconflict", "dynamic", ["temporal_validity"], states=[{
            "interaction_id": "i1", "valid_from": None, "valid_until": None, "missing_fields": [],
        }]
    )])
    metrics = summary["structural"]
    labels = summary["per_example"][0]["reconstruction_error_labels"]
    assert metrics["relevant_field_denominator"] == 1
    assert metrics["relevant_field_missing_rate"] == 1
    assert "missing_temporal_state" in labels


def test_valid_until_missing_remains_schema_diagnostic_only():
    summary = evaluate_records([_record(
        "open-ended", "memconflict", "dynamic", ["temporal_validity"], states=[{
            "interaction_id": "i1", "valid_from": "2024-01-01", "valid_until": None,
            "missing_fields": ["valid_until"],
        }]
    )])
    metrics = summary["structural"]
    assert metrics["relevant_field_missing_rate"] == 0
    assert metrics["schema_missing_field_rates"]["valid_until"] == 1
    assert "missing_temporal_state" not in summary["per_example"][0]["reconstruction_error_labels"]


def test_visible_interaction_id_validity_is_not_semantic_binding_correctness():
    record = _record("visible-wrong-target", "gatemem", "privacy", ["authorization_scope"], states=[{
        "interaction_id": "i2", "authorized_scopes": ["employee"], "missing_fields": [],
    }])
    summary = evaluate_records([record])
    metrics = summary["structural"]
    assert metrics["grounding_metric"] == "visible_history_interaction_id_validity"
    assert metrics["grounding_coverage"] == 1
    assert "wrong_or_unsupported_grounding" not in summary["per_example"][0]["reconstruction_error_labels"]
