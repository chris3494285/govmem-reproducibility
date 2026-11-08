# OAKS-derived staged artifacts

This inventory identifies every staged `data/` or `results/` file that contains
OAKS-derived aggregate labels or values. Raw OAKS question, answer, context, or
per-example material is not included.

The release conservatively treats these aggregate files as derivative works
under the Adobe Research License. The package therefore:

1. includes the verbatim Adobe Research License at
   `third_party/licenses/Adobe-Research-License-OAKS-Novel.txt`;
2. restricts use and redistribution to noncommercial research purposes; and
3. retains the upstream ownership and attribution statements.

## Metadata

- `data/final_main_governance_base_manifest_v2.freeze.json`

## Final-main and leakage-repair aggregates

- `results/answer_leakage_repair/final_v3_analysis/FINALIZATION_REPORT.md`
- `results/answer_leakage_repair/final_v3_analysis/control_plane_propagation/mechanism_by_source.csv`
- `results/answer_leakage_repair/final_v3_analysis/control_plane_propagation/mechanism_summary.json`
- `results/answer_leakage_repair/final_v3_analysis/cross_provider_summary.json`
- `results/answer_leakage_repair/final_v3_analysis/governance_evidence_audit/governance_evidence_by_source.csv`
- `results/answer_leakage_repair/final_v3_analysis/governance_evidence_audit/governance_evidence_schema_notes.md`
- `results/answer_leakage_repair/final_v3_analysis/governance_evidence_audit/governance_evidence_summary.json`
- `results/answer_leakage_repair/final_v3_analysis/licensing_blocker_report.json`
- `results/answer_leakage_repair/final_v3_analysis/licensing_blocker_report.md`
- `results/answer_leakage_repair/final_v3_analysis/matched_fault_support/matched_fault_by_source.csv`
- `results/answer_leakage_repair/final_v3_analysis/matched_fault_support/matched_fault_overlap_counts.csv`
- `results/answer_leakage_repair/final_v3_analysis/matched_fault_support/matched_fault_summary.json`
- `results/answer_leakage_repair/final_v3_analysis/old_vs_new_statistics.csv`
- `results/answer_leakage_repair/final_v3_analysis/old_vs_new_statistics.json`
- `results/answer_leakage_repair/final_v3_analysis/paper_update_checklist.md`
- `results/answer_leakage_repair/final_v3_analysis/provider_summaries.json`
- `results/answer_leakage_repair/final_v3_analysis/providers/anthropic/analysis_summary.json`
- `results/answer_leakage_repair/final_v3_analysis/providers/anthropic/analysis_table.csv`
- `results/answer_leakage_repair/final_v3_analysis/providers/deepseek/analysis_summary.json`
- `results/answer_leakage_repair/final_v3_analysis/providers/deepseek/analysis_table.csv`
- `results/answer_leakage_repair/final_v3_analysis/providers/gemini/analysis_summary.json`
- `results/answer_leakage_repair/final_v3_analysis/providers/gemini/analysis_table.csv`
- `results/answer_leakage_repair/final_v3_analysis/providers/grok/analysis_summary.json`
- `results/answer_leakage_repair/final_v3_analysis/providers/grok/analysis_table.csv`
- `results/answer_leakage_repair/final_v3_analysis/providers/kimi/analysis_summary.json`
- `results/answer_leakage_repair/final_v3_analysis/providers/kimi/analysis_table.csv`
- `results/answer_leakage_repair/final_v3_analysis/providers/openai/analysis_summary.json`
- `results/answer_leakage_repair/final_v3_analysis/providers/openai/analysis_table.csv`
- `results/final_main_cross_provider_summary_v3/cross_provider_summary.csv`
- `results/final_main_cross_provider_summary_v3/cross_provider_summary.json`
- `results/final_main_cross_provider_summary_v3/cross_provider_summary.md`

## B4 validation aggregates

- `results/b4_scoring_validation/JUDGE_PROTOCOL_AUDIT.md`
- `results/b4_scoring_validation/b4_validation_summary.md`
- `results/b4_scoring_validation/oaks_input_summary.json`
- `results/b4_scoring_validation/paper_impact_audit/PAPER_SCORER_IMPACT_AUDIT.md`
- `results/b4_scoring_validation/paper_impact_audit/paper_impact_summary.json`
- `results/b4_scoring_validation/paper_impact_audit/remediation_options.json`

## Formal, observability, and evidence aggregates

- `results/formal_300_subgroup_v1/FORMAL_300_SUBGROUP_REPORT.md`
- `results/formal_300_validation_v1/FORMAL_300_VALIDATION_PROTOCOL.md`
- `results/jss_rq1_observability_v2/observability_by_source.csv`
- `results/jss_rq1_observability_v2/observability_limitations.md`
- `results/paper_evidence_audit/condition_trace_validation.json`
- `results/paper_evidence_audit/condition_trace_validation.md`
- `results/paper_evidence_audit/event_macro_reconciliation.md`

## Source/event/fault aggregates

- `results/source_event_fault_distribution_v1/SOURCE_EVENT_FAULT_DISTRIBUTION_REPORT.md`
- `results/source_event_fault_distribution_v1/SOURCE_EVENT_FAULT_PAPER_TABLE.tex`
- `results/source_event_fault_distribution_v1/distribution_summary.json`
- `results/source_event_fault_distribution_v1/source_event_counts.csv`
- `results/source_event_fault_distribution_v1/source_event_fault_counts.csv`
- `results/source_event_fault_distribution_v1/source_fault_counts.csv`

## Files that reference but do not contain OAKS data

Original implementation, configuration, tests, acquisition instructions, and
the license/notice index also mention OAKS by name. They do not contain copied
OAKS records or OAKS-derived values and are not listed as data artifacts here.
They remain covered by the applicable original-code or documentation scope.
