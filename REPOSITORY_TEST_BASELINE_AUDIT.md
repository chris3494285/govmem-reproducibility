# Repository pytest baseline audit

Audit date: 2026-10-05  
Repository: `<PUBLIC_REPO_ROOT>`  
Mode: offline, read-only inspection except for this report  
API/provider/model calls: **0**

## Executive result

The current full-suite baseline has **52 failing tests**. All 52 collect and
start normally. Every observed failure is caused by a declared file dependency
that is absent from this public-release working tree; none currently provides
evidence of a code regression.

The count is based on the current local tree, which has an ignored local symlink
for `data/final_main_governance_condition_manifest_v2.jsonl` pointing to the
exact frozen local artifact (SHA-256
`58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8`).
That text-bearing manifest is not part of the public release. Without a local
copy, additional artifact-integration tests fail earlier.

| Classification | Failing tests | Meaning |
|---|---:|---|
| `SELF_CONTAINED_CODE_FAILURE` | 0 | No self-contained test currently fails. |
| `MISSING_PUBLIC_RELEASE_ARTIFACT` | 18 | Non-secret frozen configuration/protocol files referenced by shipped code and tests are absent. |
| `MISSING_PRIVATE_OR_HISTORICAL_ARTIFACT` | 34 | The test requires licensed text, private provider output, superseded experiment state, or an explicitly omitted historical analysis artifact. |
| `REAL_CODE_REGRESSION` | 0 | No failure reached an assertion demonstrating changed code behavior. |
| `UNKNOWN` | 0 | Every failure has a concrete missing-path explanation. |

This classification does **not** say that all absent files should be published.
In particular, text-bearing benchmark manifests and raw provider outputs should
remain private unless licensing and release review explicitly permit them.

## Missing-artifact inventory

“Offline regeneration” means reconstruction by an existing deterministic
pipeline, not fabrication. “Conditional” means the pipeline is offline but its
required historical/private inputs are also absent from this public tree.

| Exact expected path(s) | Current equivalent or canonical artifact | Expected public? | Deterministically regenerable offline? | Classification |
|---|---|---:|---|---|
| `data/final_main_governance_condition_manifest_v3.jsonl` | Freeze/provenance: `data/final_main_governance_condition_manifest_v3.freeze.json`; public scores: `results/final_main_sanitized/*.jsonl` | No; `ARTIFACTS.md` explicitly says the text-bearing v3 manifest is excluded pending licensing review | Conditional: `scripts/repair_answer_leakage.py` can rebuild it only with the local v2 manifest and official LongMemEval source | `MISSING_PRIVATE_OR_HISTORICAL_ARTIFACT` |
| `results/final_main_deepseek_v4_pro/scored_results.jsonl` | `results/final_main_sanitized/deepseek.jsonl` | Raw/private path: no; sanitized equivalent: yes | No from public data as the original private payload cannot be reconstructed; score-bearing analysis can use the sanitized artifact if explicitly supported | `MISSING_PRIVATE_OR_HISTORICAL_ARTIFACT` |
| `results/final_main_openai_gpt56_sol_final_v2/scored/full_results_scored.jsonl` | `results/final_main_sanitized/openai.jsonl` | Raw/private path: no; sanitized equivalent: yes | Same as above | `MISSING_PRIVATE_OR_HISTORICAL_ARTIFACT` |
| `results/final_main_anthropic_sonnet5_final/scored/full_results_scored.jsonl` | `results/final_main_sanitized/anthropic.jsonl` | Raw/private path: no; sanitized equivalent: yes | Same as above | `MISSING_PRIVATE_OR_HISTORICAL_ARTIFACT` |
| `results/final_main_gemini_3_flash_final/scored/full_results_scored.jsonl` | `results/final_main_sanitized/gemini.jsonl` | Raw/private path: no; sanitized equivalent: yes | Same as above | `MISSING_PRIVATE_OR_HISTORICAL_ARTIFACT` |
| `results/final_main_kimi_k2_6_nothinking_final/scored/full_results_scored.jsonl` | `results/final_main_sanitized/kimi.jsonl` | Raw/private path: no; sanitized equivalent: yes | Same as above | `MISSING_PRIVATE_OR_HISTORICAL_ARTIFACT` |
| `results/final_main_anthropic_sonnet5/truncated_condition_ids.txt`; `results/final_main_anthropic_sonnet5/nontrunc_failure_condition_ids.txt` | No current canonical ID-list artifact; current sanitized Anthropic rows retain truncation/failure metadata sufficient for some audits | Historical execution metadata; not required as a public primary result | Conditional: derivable only from the exact historical Anthropic result artifact and frozen classification rule | `MISSING_PRIVATE_OR_HISTORICAL_ARTIFACT` |
| `configs/final_main_token_budget_anthropic_repair_v2.json` | `configs/final_main_token_budget_v2.json` is the main budget, not an equivalent repair contract | Yes: small non-secret frozen protocol metadata referenced by public loader/tests | No declared generator; exact file is recoverable from repository history/local development provenance | `MISSING_PUBLIC_RELEASE_ARTIFACT` |
| `configs/final_main_token_budget_gemini_v1.json` | `configs/final_main_token_budget_gemini_flash_v1.json` is a different schema/contract | Yes | No declared generator; restore exact historical artifact rather than reconstructing it by hand | `MISSING_PUBLIC_RELEASE_ARTIFACT` |
| `configs/final_main_token_budget_gemini_flash_repair_v1.json`; `configs/final_main_token_budget_gemini_flash_repair_v2.json` | No equivalent current repair contract | Yes | No declared generator; v2 is recoverable from repository history, and both exist in the local development provenance | `MISSING_PUBLIC_RELEASE_ARTIFACT` |
| `configs/final_main_token_budget_kimi_v1.json`; `configs/final_main_token_budget_kimi_v2.json`; `configs/final_main_token_budget_kimi_v3.json`; `configs/final_main_token_budget_kimi_repair_v1.json` | `configs/final_main_token_budget_kimi_v4.json` is current but intentionally not equivalent to v1/v2/v3/repair-v1 | Yes | No declared generator; exact historical contracts must be restored, not inferred from v4 | `MISSING_PUBLIC_RELEASE_ARTIFACT` |
| `data/final_main_governance_condition_manifest_v1.jsonl` | v2 freeze metadata and a local ignored v2 manifest exist; v1 is a distinct parent | Historical text-bearing parent, not needed as a current public result | Conditional: exact v1 exists in historical repository provenance; rebuilding scientifically requires its original source inputs | `MISSING_PRIVATE_OR_HISTORICAL_ARTIFACT` |
| `data/final_main_governance_smoke_base_manifest_v1.jsonl`; `data/final_main_governance_smoke_condition_manifest_v1.jsonl` | `data/final_main_governance_smoke_condition_manifest_v2.jsonl` is current but not byte-equivalent | Historical fixtures | Conditional: recoverable from historical provenance; not derivable from v2 without the frozen v1 selection record | `MISSING_PRIVATE_OR_HISTORICAL_ARTIFACT` |
| `results/final_main_cross_provider_summary_v1/cross_provider_summary.json`; `results/final_main_cross_provider_summary_v2/cross_provider_summary.json` | `results/final_main_cross_provider_summary_v3/cross_provider_summary.json` and `results/answer_leakage_repair/final_v3_analysis/cross_provider_summary.json` supersede them | Historical, not current canonical outputs | Conditional: reproducible only with the historical provider analyses/results; exact files are recoverable from historical provenance | `MISSING_PRIVATE_OR_HISTORICAL_ARTIFACT` |
| `results/final_main_deepseek_v4_pro_vs_openai_gpt56_sol_analysis/analysis_summary.json`; `results/final_main_deepseek_v4_pro_vs_anthropic_sonnet5_analysis/analysis_summary.json`; `results/final_main_deepseek_v4_pro_vs_gemini_3_flash_analysis/analysis_summary.json`; `results/final_main_gemini_3_flash_vs_kimi_k2_6_analysis/analysis_summary.json` | Current six-provider v3 analyses live under `results/answer_leakage_repair/final_v3_analysis/providers/` | Historical only | Conditional on historical score artifacts | `MISSING_PRIVATE_OR_HISTORICAL_ARTIFACT` |
| `results/final_main_kimi_k2_6_nothinking_final/full_results_merged.jsonl`; `results/final_main_grok_43_full/full_results.jsonl` | Sanitized Kimi/Grok score files exist; Grok scored output also exists at its historical path | Raw answer-bearing artifacts should not be required publicly | No from sanitized outputs | `MISSING_PRIVATE_OR_HISTORICAL_ARTIFACT` |
| `data/main_governance_base_manifest.jsonl`; `results/oaks_novel_main_dataset_audit/oaks_novel_candidates.jsonl` | Current final base cohort: `data/final_main_governance_base_manifest_v2.jsonl`; no equivalent public candidate-pool audit | Historical/source-audit inputs; candidate text may be licensed | Conditional: `scripts/build_final_main_governance_manifest.py` is offline, but both source inputs are required | `MISSING_PRIVATE_OR_HISTORICAL_ARTIFACT` |
| `results/system_validation_statistics/system_validation_statistics_by_system.csv`; `results/system_validation_statistics/system_validation_statistics_pooled.csv`; `results/system_validation_statistics/system_validation_mcnemar.csv`; `results/system_validation_analysis/system_validation_scoring_coverage.csv` | No equivalent current files in this tree; `ARTIFACTS.md` states this separate experiment is not included in the clean branch | No for this public-release baseline as currently documented | Conditional on the omitted formal system-validation outputs; no model calls are needed once those inputs exist | `MISSING_PRIVATE_OR_HISTORICAL_ARTIFACT` |

The historical cross-provider summary code also declares the raw/analysis paths
listed above. The current immediate failure occurs on the missing v1/v2 summary,
but restoring only that first file would expose the remaining historical
dependencies; therefore they are included in this inventory.

## Classification of every failing test

### `MISSING_PUBLIC_RELEASE_ARTIFACT` — 18

These tests exercise shipped protocol loaders/runners using small, non-secret
configuration files. Their absence is a release-packaging gap, not evidence that
the loader or runner is wrong.

1. `tests/test_final_main_anthropic_runner.py::test_repair_budget_is_rejected_for_normal_full_run` — missing `configs/final_main_token_budget_anthropic_repair_v2.json`.
2. `tests/test_final_main_kimi.py::test_kimi_full_dry_run_is_offline` — missing Kimi v1 budget config.
3. `tests/test_final_main_kimi.py::test_kimi_budget_config_is_strict` — missing Kimi v1 budget config.
4. `tests/test_final_main_kimi.py::test_kimi_v2_budget_config_has_exact_canary_repair_ceilings` — missing Kimi v2 budget config.
5. `tests/test_final_main_kimi.py::test_kimi_model_parameters_are_deterministic` — missing default Kimi v1 budget config.
6. `tests/test_final_main_kimi.py::test_kimi_disabled_thinking_is_a_uniform_dry_run_setting` — missing default Kimi v1 budget config.
7. `tests/test_final_main_kimi.py::test_kimi_provider_default_temperature_remains_one` — missing default Kimi v1 budget config.
8. `tests/test_final_main_kimi.py::test_kimi_v2_targeted_dry_run_is_offline_and_resolves_budget` — missing Kimi v2 budget config.
9. `tests/test_final_main_kimi.py::test_kimi_v3_budget_config_is_strict_and_v2_remains_unchanged` — missing Kimi v3 (and subsequently v2) configs.
10. `tests/test_final_main_kimi.py::test_kimi_runner_rejects_unknown_budget_schema` — missing Kimi v3 source config used to construct the invalid fixture.
11. `tests/test_final_main_kimi.py::test_kimi_repair_v1_budget_config_is_strict_and_formal_v4_unchanged` — missing Kimi repair-v1 config.
12. `tests/test_final_main_kimi.py::test_kimi_repair_dry_run_is_offline` — missing Kimi repair-v1 config.
13. `tests/test_final_main_token_budget.py::test_frozen_loader_rejects_anthropic_repair_config` — missing Anthropic repair-v2 config.
14. `tests/test_final_main_token_budget.py::test_gemini_measurement_loader_accepts_only_explicit_gemini_ceilings` — missing Gemini v1 config.
15. `tests/test_final_main_token_budget.py::test_gemini_flash_repair_loader_accepts_only_repair_ceilings` — missing Gemini Flash repair-v1 config.
16. `tests/test_final_main_token_budget.py::test_gemini_flash_repair_loader_rejects_unauthorized_ceiling_change` — missing Gemini Flash repair-v1 source config.
17. `tests/test_final_main_token_budget.py::test_gemini_flash_repair_v2_loader_accepts_only_v2_ceilings` — missing Gemini Flash repair-v2 config.
18. `tests/test_final_main_token_budget.py::test_anthropic_repair_loader_accepts_only_the_explicit_repair_contract` — missing Anthropic repair-v2 config.

### `MISSING_PRIVATE_OR_HISTORICAL_ARTIFACT` — 34

1. `tests/test_answer_leakage.py::test_corrected_v3_manifest_has_no_leakage_or_unresolved_repair_provenance` — intentionally excluded licensed/text-bearing v3 manifest.
2. `tests/test_answer_leakage.py::test_release_guard_rejects_direct_oracle_provenance` — same v3 manifest dependency.
3. `tests/test_control_plane_propagation.py::test_control_plane_analysis_is_complete_and_offline` — historical private provider score paths; first missing path is DeepSeek.
4. `tests/test_control_plane_propagation.py::test_analysis_is_deterministic` — same provider score set.
5. `tests/test_control_plane_propagation.py::test_missing_or_duplicate_result_join_fails` — directly reads the missing historical DeepSeek score artifact to construct a negative fixture.
6. `tests/test_final_main_anthropic_runner.py::test_anthropic_repair_selects_exact_truncated_ids_and_uses_repair_budgets` — missing historical truncated/non-truncated ID cohorts (and repair config).
7. `tests/test_final_main_condition_manifest_v2.py::test_parent_artifacts_and_hashes_are_unchanged` — missing v1 condition parent.
8. `tests/test_final_main_condition_manifest_v2.py::test_smoke_v2_is_same_base_subset_and_complete` — missing v1 smoke base manifest.
9. `tests/test_final_main_condition_manifest_v2_builder.py::test_two_temporary_builds_are_byte_identical` — builder requires missing v1 condition parent.
10. `tests/test_final_main_cross_provider_summary.py::test_final_main_summary_has_five_formal_providers_and_frozen_kimi_values` — missing historical v1 summary, followed by historical analyses/raw results.
11. `tests/test_final_main_cross_provider_summary.py::test_prior_four_model_rows_are_unchanged` — missing historical v1 summary.
12. `tests/test_final_main_cross_provider_summary.py::test_provider_effect_directions_and_stale_fault_order` — missing historical v1 summary.
13. `tests/test_final_main_cross_provider_summary.py::test_final_main_summary_generation_is_deterministic` — missing historical v1 summary.
14. `tests/test_final_main_cross_provider_summary.py::test_final_main_v3_has_six_providers_and_grok_values` — v3 builder first depends on the missing historical v1/v2 chain.
15. `tests/test_final_main_cross_provider_summary.py::test_final_main_v3_preserves_prior_five_and_is_deterministic` — directly requires historical v2 summary.
16. `tests/test_final_main_governance_manifest.py::test_final_manifest_has_731_bases_and_preserves_clear_cohort` — missing historical clear cohort and OAKS candidate audit.
17. `tests/test_final_main_governance_manifest.py::test_oaks_selection_is_balanced_and_deterministic` — directly reads the missing OAKS candidate audit.
18. `tests/test_final_main_governance_manifest.py::test_support_reasons_and_no_legacy_reuse` — cascades because the prior build could not create `/tmp/final_main_governance_test.jsonl`; the underlying blockers are the same historical inputs.
19. `tests/test_final_main_governance_manifest.py::test_manifest_hash_is_stable` — same cascade from the failed historical build.
20. `tests/test_final_main_smoke_deepseek_runner.py::test_dry_run_validates_frozen_manifest_without_provider` — missing v1 smoke condition manifest.
21. `tests/test_final_main_smoke_deepseek_runner.py::test_prompt_is_deterministic_and_condition_specific` — same missing v1 smoke manifest.
22. `tests/test_final_main_smoke_deepseek_runner.py::test_mocked_success_records_parse_and_telemetry` — same missing v1 smoke manifest.
23. `tests/test_final_main_smoke_deepseek_runner.py::test_resume_reuses_successes_and_rejects_identity_mismatch` — same missing v1 smoke manifest.
24. `tests/test_final_main_smoke_deepseek_runner.py::test_no_api_key_is_written_to_dry_run` — same missing v1 smoke manifest.
25. `tests/test_final_main_smoke_deepseek_runner.py::test_malformed_json_then_valid_response_recovers` — same missing v1 smoke manifest.
26. `tests/test_final_main_smoke_deepseek_runner.py::test_provider_transport_error_then_valid_response_recovers` — same missing v1 smoke manifest.
27. `tests/test_final_main_smoke_deepseek_runner.py::test_three_malformed_responses_persist_final_failure` — same missing v1 smoke manifest.
28. `tests/test_final_main_smoke_deepseek_runner.py::test_retry_budget_and_prompt_are_invariant` — same missing v1 smoke manifest.
29. `tests/test_matched_fault_support.py::test_matched_support_analysis_uses_same_bases_and_is_deterministic` — historical private provider score set.
30. `tests/test_matched_fault_support.py::test_stale_overlap_counts_are_transparent` — same provider score set.
31. `tests/test_sanitize_final_results.py::test_sanitizer_is_deterministic_and_excludes_provider_payloads` — sanitizer intentionally needs raw/private provider score files; already-sanitized public files cannot prove raw-field exclusion.
32. `tests/test_system_validation_paper_artifacts.py::test_paper_tables_preserve_statistics_and_coverage` — separate formal system-validation statistics omitted from this clean branch.
33. `tests/test_system_validation_paper_artifacts.py::test_paper_artifact_files_and_cautious_text` — same omitted statistics/coverage inputs.
34. `tests/test_system_validation_paper_artifacts.py::test_paper_builder_does_not_need_inference_or_provider` — same omitted statistics/coverage inputs.

### Other categories

- `SELF_CONTAINED_CODE_FAILURE`: none.
- `REAL_CODE_REGRESSION`: none demonstrated by the current failures.
- `UNKNOWN`: none.

## Genuinely actionable issues

There are no demonstrated code regressions in this failure set. The actionable
issues are release/test-boundary issues:

1. **Restore or deliberately retire eight public protocol configs.** The
   Anthropic, Gemini, and Kimi historical budget contracts contain no benchmark
   text or provider secrets and are referenced by shipped loaders/tests. Exact
   provenance copies should be restored after hash review; they should not be
   recreated from memory.
2. **Stop treating private/historical integrations as default tests.** Tests
   requiring licensed manifests, raw provider outputs, or omitted historical
   experiment trees should declare those dependencies and run only when the
   artifacts are explicitly supplied.
3. **Add public-canonical integration coverage.** Control-plane and matched-fault
   scripts currently default to old private paths even though canonical public
   sanitized score artifacts exist. A future, separately reviewed change could
   accept explicit sanitized inputs and verify hashes. This audit does not claim
   they are interchangeable for every use (notably, sanitizer tests require raw
   inputs).
4. **Remove order dependence in two historical manifest tests.** The support and
   hash tests consume `/tmp/final_main_governance_test.jsonl` created by an
   earlier test. When retained as artifact integration tests, each test should
   build its own temporary output after dependency checks. This is a test
   robustness issue, not the current root failure.

## Tests that cannot run from the public release alone

The following capabilities require artifacts intentionally or historically
absent from this tree:

- direct audit of the licensed/text-bearing v3 condition manifest;
- reconstruction of v1/v2 manifest ancestry and smoke-cohort byte identity;
- raw-provider sanitization verification;
- historical v1/v2 cross-provider summary preservation;
- historical Anthropic truncation-repair cohort reproduction;
- old governance candidate-pool selection reconstruction;
- the separate formal system-validation paper-artifact build;
- analyses hard-wired to old private provider score directories.

These should not be made green by fabricated placeholders, weakened counts, or
substitution of a semantically different current artifact.

## Proposed pytest marker structure (not implemented)

```ini
markers =
    unit: self-contained deterministic tests requiring no repository result artifacts
    b4: offline B4 protocol, safety, lifecycle, and analysis tests
    artifact_integration: tests requiring public frozen artifacts with verified hashes
    private_artifact: tests requiring licensed text, raw provider output, or intentionally local historical artifacts
```

Recommended assignment:

- `unit`: core schemas, fault operators, scoring primitives, provider payload
  construction with mocks, parser behavior, retry logic, and the restored
  final-main compatibility helpers.
- `b4`: `tests/test_b4_validation.py` and other B4-only offline invariants.
- `artifact_integration`: current public sanitized results, freeze metadata,
  current public summaries, and the small historical budget configs once those
  configs are restored and hash-reviewed.
- `private_artifact`: v1/v2 ancestry checks, text-bearing v3 audit, raw sanitizer
  inputs, old provider-result analyses, and omitted system-validation outputs.

Tests may carry more than one marker (for example, `b4` plus
`artifact_integration`). A private-artifact test should fail clearly when
explicitly selected without its input; it should not fabricate or silently
download data.

## Minimal scientifically defensible default suite

The default CI/test command should run all self-contained tests plus B4 tests
and only those integration tests whose required public artifacts are actually
distributed and hash-verified:

```bash
python -m pytest -q -m "unit or b4 or (artifact_integration and not private_artifact)"
```

Until markers are applied, the defensible explicit baseline is:

```bash
python -m pytest -q \
  tests/test_final_main_gemini_runner.py \
  tests/test_final_main_openai_runner.py \
  tests/test_final_main_recovery_tools.py \
  tests/test_b4_validation.py
```

This is intentionally a minimum, not a proposal to weaken scientific checks.
Private/historical integration tests should remain intact and run in an
artifact-complete environment. The public default should accurately represent
what the release actually contains.

## Offline commands used

```bash
python -m pytest -q
python -m pytest -q tests/test_answer_leakage.py \
  tests/test_control_plane_propagation.py \
  tests/test_final_main_anthropic_runner.py \
  tests/test_final_main_condition_manifest_v2.py \
  tests/test_final_main_condition_manifest_v2_builder.py \
  tests/test_final_main_cross_provider_summary.py \
  tests/test_final_main_governance_manifest.py --tb=short
python -m pytest -q tests/test_final_main_kimi.py \
  tests/test_final_main_smoke_deepseek_runner.py \
  tests/test_final_main_token_budget.py \
  tests/test_matched_fault_support.py \
  tests/test_sanitize_final_results.py \
  tests/test_system_validation_paper_artifacts.py --tb=short
rg --files results configs data
git log --all -- <missing-path>
git -C <PRIVATE_REPO_ROOT> ls-files
```

All commands were offline. No result, scorer, B4, benchmark, or test file was
modified during this audit.
