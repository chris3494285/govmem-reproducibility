# Native-evidence v3 rerun execution plan

The v3 manifest has 4,891 unique conditions and SHA-256 `4b84068a341d912f43d07c9d92059f75224ea2636abf69e40ba6e71a0a1144cb`. The repair cohort is 265 conditions (clean 53, corrupted 106, enforced 106), all LongMemEval temporal rows. Each provider dry run completed with `api_calls_made: 0`; the commands and token estimates are recorded in `rerun_cost_estimate.json`.

Recommended order is DeepSeek and OpenAI first (smallest output ceilings), then Anthropic and Grok, then Kimi and Gemini. Monetary estimates are unresolved unless a local runner price was already configured; no external pricing source was queried. The Anthropic and Gemini figures are local upper-bound defaults, not billing claims.

The existing runners support `--resume`. The shared runner persists completed rows and only reuses a cached row when its model, manifest hash, prompt hash, token budget, and model parameters match. OpenAI batch lifecycle is supported through its existing `--execution-mode batch`, `--batch-action submit|status|retrieve` flow; use the documented runner rather than manually constructing requests.

After each provider completes, validate its 265 rows with the provider validation command in the JSON report, then merge with the corresponding `merge_command`. The merge excludes exactly the 265 changed IDs from the old sanitized baseline, validates against v3, and therefore preserves the other 4,626 conditions. Sanitize merged outputs before public distribution if they contain provider response metadata.

## Post-inference offline commands

Run `python scripts/audit_answer_leakage.py` first. Then run the six-provider offline analyses using the merged, sanitized condition-level files and v3:

```bash
python scripts/analyze_final_main_results.py --run-a results/answer_leakage_repair/merged_PROVIDER.jsonl --run-b results/final_main_PROVIDER_independent_replication.jsonl --manifest data/final_main_governance_condition_manifest_v3.jsonl --base-manifest data/final_main_governance_base_manifest_v2.jsonl --analysis-spec configs/final_main_analysis_spec_v1.json --failure-spec configs/final_main_failure_handling_spec_v1.json --output-dir results/answer_leakage_repair/analysis_PROVIDER
python scripts/analyze_control_plane_propagation.py --manifest data/final_main_governance_condition_manifest_v3.jsonl --output-dir results/answer_leakage_repair/control_plane_propagation
python scripts/analyze_matched_fault_support.py --manifest data/final_main_governance_condition_manifest_v3.jsonl --output-dir results/answer_leakage_repair/matched_fault_support
python scripts/summarize_rq1_cross_provider.py
python scripts/build_system_validation_paper_artifacts.py --output-dir results/answer_leakage_repair/paper_figures
python scripts/build_rq1_paper_table.py --manifest data/final_main_governance_condition_manifest_v3.jsonl --output-dir results/answer_leakage_repair/rq1
```

Replace `PROVIDER` with each provider name. `merged_PROVIDER.jsonl` is the single complete v3 file used for that provider's primary metrics. `final_main_PROVIDER_independent_replication.jsonl` must be a genuinely independent historical or rerun result with matching identity; if it does not exist, omit replication analysis rather than duplicating the primary file. Existing table and figure builders must be rerun after provider merges; no v2 result is silently relabeled as v3.

The v2 audit and all v2 scored results remain historical provenance. v3 is not submission-ready until all six providers are rerun and the downstream analyses, tables, and figures are regenerated and validated.
