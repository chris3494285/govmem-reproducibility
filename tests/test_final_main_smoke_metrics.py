from scripts.summarize_final_main_smoke import summarize


def _row(base, kind, correct, op=None):
    return {"base_id": base, "condition_type": kind, "pair_id": None if kind == "clean" else f"{base}::{op}", "fault_operator": op, "analysis_event_type": "update", "source_dataset": "longmemeval", "correctness": correct, "parse_success": True}


def test_pairwise_corruption_and_recovery_metrics():
    rows = [
        _row("b1", "clean", True), _row("b1", "corrupted", False, "stale"), _row("b1", "enforced", True, "stale"),
        _row("b2", "clean", True), _row("b2", "corrupted", True, "missing"), _row("b2", "enforced", True, "missing"),
    ]
    summary = summarize(rows)
    assert summary["condition_accuracy"]["clean"]["accuracy"] == 1.0
    assert summary["pairwise_overall"]["scored_n"] == 2
    assert summary["pairwise_overall"]["corruption_delta"]["mean"] == -0.5
    assert summary["pairwise_overall"]["recovery_delta"]["mean"] == 0.5
    assert summary["pairwise_overall"]["recovery_fraction"]["scored_n"] == 1


def test_unscored_correctness_is_not_imputed():
    rows = [_row("b1", "clean", None), _row("b1", "corrupted", None, "stale"), _row("b1", "enforced", None, "stale")]
    summary = summarize(rows)
    assert summary["condition_accuracy"]["clean"]["scored_n"] == 0
    assert summary["pairwise_overall"]["scored_n"] == 0
    assert summary["pairwise_overall"]["corruption_delta"]["mean"] is None
