import json
from pathlib import Path

from eval.dataset import load_dataset
from eval.runner import run_demo


def test_runner_produces_all_methods_and_results(tmp_path):
    dataset = Path("data/sample_longmemeval.jsonl")
    results = run_demo(dataset, results_dir=tmp_path)

    methods = {entry["method"] for entry in results["methods"]}
    assert methods == {"no_memory", "naive_memory", "warehouse_no_governance", "govmem"}

    summary_path = tmp_path / "demo_summary.md"
    results_path = tmp_path / "demo_results.json"
    traces_path = tmp_path / "demo_traces.json"
    trace_report_path = tmp_path / "demo_trace_report.md"
    assert summary_path.exists()
    assert results_path.exists()
    assert traces_path.exists()
    assert trace_report_path.exists()

    naive = next(entry for entry in results["methods"] if entry["method"] == "naive_memory")
    govmem = next(entry for entry in results["methods"] if entry["method"] == "govmem")
    assert govmem["exact_match"] >= naive["exact_match"]
    assert govmem["token_f1"] >= naive["token_f1"]
    assert govmem["policy_violation_count"] < naive["policy_violation_count"]
    assert govmem["stale_memory_error_count"] < naive["stale_memory_error_count"]
    assert govmem["safe_exact_match"] >= naive["safe_exact_match"]

    examples = load_dataset(dataset)
    traces = traces_path.read_text(encoding="utf-8")
    trace_items = json.loads(traces)
    assert len(trace_items) == len(examples) * 4

    example_ids = {example.id for example in examples}
    methods = {"no_memory", "naive_memory", "warehouse_no_governance", "govmem"}
    for example_id in example_ids:
        per_example = [item for item in trace_items if item["example_id"] == example_id]
        assert {item["method"] for item in per_example} == methods
        for item in per_example:
            assert "retrieved_memory_ids" in item
            assert "retrieved_memory_contents" in item
            assert "retrieved_memory_metadata" in item
            assert "policy_violation" in item
            assert "stale_memory_error" in item

    report = trace_report_path.read_text(encoding="utf-8")
    assert "GovMem filtered deleted/private/unauthorized" in report or "Governance and conflict resolution improved answer selection." in report

    summary = summary_path.read_text(encoding="utf-8")
    assert "## Key Takeaway" in summary
    assert "safe_exact_match" in summary
    assert "Warehouse structure alone does not eliminate unsafe or stale retrieval." in summary
    assert "Official Retrieval Sanity Check" not in summary
    assert str(tmp_path / "demo_traces.json") in summary
    assert str(tmp_path / "demo_trace_report.md") in summary
