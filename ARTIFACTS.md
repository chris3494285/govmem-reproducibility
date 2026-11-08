# GovMem paper artifact map

Hashes below are SHA-256 hashes of the files currently present. Recompute them
with `shasum -a 256` after copying an artifact. V2 is historical and
superseded; v3 is the completed native-evidence repair.

This is an **aggregate-only** release: no row-level result file is shipped. The
six condition-level score files (`results/final_main_tier_b/*.jsonl` and the
superseded `results/final_main_sanitized/*.jsonl`), the per-instance/per-base
tables, the cohort/judge-request/validation dumps, the formal-300 execution
plan, and the clean-repeat generation/judging ledgers are all excluded, as are
every source-derived benchmark identifier and per-row content fingerprint.
Only aggregate tables and summaries remain; the reproducibility tiers are in
`docs/REPRODUCTION_TIERS.md`.

Text-bearing v3 manifests and provider rerun manifests are intentionally not
redistributed pending license review; obtain LongMemEval, Memora, and OAKS
Novel from their official sources. Final headline values are corruption
−27.55 pp, recovery +27.51 pp, context-changed corruption −26.04 pp,
context-unchanged corruption −1.28 pp, control-plane change 96.25%, and
restoration 100%.

| Paper claim/table/figure | Generating script | Input files | Output files | Expected counts | Expected SHA-256 | API? |
|---|---|---|---|---:|---|---|
| Frozen base cohort and 191/380/160 split | `scripts/build_final_main_governance_manifest_v2.py` | frozen parent manifest and source-audit artifacts | `data/final_main_governance_base_manifest_v2.jsonl` and freeze metadata | 731 bases | `a2601a7ecb46034faf5dcc7ad00b5a851e19ae01561876fa681fd18af168ca79` | No |
| Frozen clean/corrupted/enforced cohort | `scripts/build_final_main_condition_manifest_v2.py` | frozen base/condition parents, source audits, `configs/main_analysis_spec_v1.json` | `data/final_main_governance_condition_manifest_v2.jsonl` and freeze metadata | 731 / 2,080 / 2,080 / 4,891 | `58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8` | No |
| Answer-leakage validity gate | `scripts/audit_answer_leakage.py` | frozen v2 condition manifest | `results/answer_leakage_audit.json` | 53 affected bases; 159 rows | `d37f6fac8ce31a22b5f974bd76ca80957f4b08f7e6cf186de3dd48649195f5fe` | No |
| Native-evidence v3 repair | `scripts/repair_answer_leakage.py` | official local LongMemEval oracle JSON; frozen v2 manifest | v3 manifest and freeze metadata remain local; public release contains only hashes/provenance after license review | 53 bases; 265 changed rows; 1,590 requests completed | `4b84068a341d912f43d07c9d92059f75224ea2636abf69e40ba6e71a0a1144cb` (local v3 manifest); `12b9a27585e9e40af8666cfef2573e8492aa7bfa504a10a85b81e798be12b941` (freeze) | No |
| V2-to-v3 repair comparison | `scripts/repair_answer_leakage.py` | v2 manifest and official native source | `results/answer_leakage_repair/changed_conditions.jsonl`, `changed_condition_ids.txt` (**excluded**, row-level) and `repair_summary.json` (shipped) | 265 rows; 265 prompts | `3623915d3fe43275dc427a234c396e07d17c50416ce4e29b1d122eb6d01c51e4` (changed rows); `3ecd6cc677597b0cecce521e5353d0c7e677bcd82755ed068eb02ef6da98573` (summary) | No |
| Corrected-source provenance | `scripts/repair_answer_leakage.py` | official native LongMemEval source | `results/answer_leakage_repair/source_evidence_provenance.json` (**excluded**) | 53 corrected bases | `4e6ba11a2ded98dc12ce6432b3719a317b0d8e892da88ac4852f93558a1774af` | No |
| Provider rerun manifests | `scripts/repair_answer_leakage.py` | corrected v3 manifest and changed prompt IDs | local-only text-bearing manifests; not redistributed | 265 conditions/provider; 1,590 total | `13156554eb6d688fd3e6b7c870bc4f2543cafdd0f8feb1fc621a73551f03c0ab` (each local file) | Yes when executed |
| Corrected v3 leakage audit | `scripts/audit_answer_leakage.py` | corrected v3 manifest | `results/answer_leakage_repair/answer_leakage_audit_v3.json` | PASS; 0 affected rows | `b77fc0b9f48ea05f33fb4ef60f83d220e5ee6678a85eb36269e2c526c0526caf` | No |
| Governance evidence audit | `scripts/audit_governance_evidence.py` | local v2 base and v3 condition manifests | `results/answer_leakage_repair/final_v3_analysis/governance_evidence_audit/` | PASS; 731 fully supported; 0 unsupported | `3224f5a3769009311c4d91d666d2e1fef1b2f1b1b1691a201e7055ff8a95222c` (summary) | No |
| Six-provider v3 main analysis | `scripts/finalize_v3_offline.py`, `scripts/analyze_final_main_results.py` | six promoted sanitized score files and locally obtained v3 manifest | `results/answer_leakage_repair/final_v3_analysis/providers/` and `cross_provider_summary.json` | 6 × 4,891 rows; clean/corrupted/enforced = 731/2,080/2,080 | `929e213cdaeb7a29a8725be026651fcffffc8e83c12b51df8a8c373ebcff7e9b` (summary) | No |
| V2-versus-v3 paper statistics | `scripts/finalize_v3_offline.py` | historical summary and six v3 primary summaries | `results/answer_leakage_repair/final_v3_analysis/old_vs_new_statistics.{json,csv,md}` | 282 statistics; 126 provider changes | `ff49e25831056d0840e0941f7adb5796a86422c69038759fe3e12054ebf717e9` (JSON) | No |
| RQ1 causal effects | `scripts/build_rq1_paper_table.py` and RQ1 analysis scripts | frozen RQ1 manifests and scored outputs | `results/rq1/`, `results/rq1_paper/` | paired provider/fault effects | per-file hashes | No |
| RQ2 fault severity | `scripts/analyze_rq2_fault_severity.py` | final scored condition outputs and frozen manifest | `results/rq2/` | supported matched fault comparisons | per-file hashes | No |
| RQ3 event sensitivity | `scripts/analyze_rq3_error_propagation.py` | frozen RQ1 artifacts | `results/rq3/` | event/fault propagation tables | per-file hashes | No |
| RQ4 source/provider heterogeneity | `scripts/analyze_formal_300_subgroups.py` and `scripts/analyze_final_main_results.py` | final Formal-300 aggregate input and six-provider final scores | `results/formal_300_subgroup_v1/`, `results/final_main_cross_provider_summary_v3/` | 298 triplets, 238 resolved base clusters, source/provider breakdowns | per-file hashes | No |
| Enforcement recovery | `scripts/analyze_final_main_results.py` | clean/corrupted/enforced score files | cross-provider summaries and RQ tables | 2,080 enforced rows | per-file hashes | No |
| Control-plane propagation | `scripts/analyze_control_plane_propagation.py` via `scripts/finalize_v3_offline.py` | six promoted score files and locally obtained v3 condition manifest | `results/answer_leakage_repair/final_v3_analysis/control_plane_propagation/` | denominator 12,480; change 0.9625; restoration 1.0 | `f3ac8564301b137bd3f2f08fe8a5029679c1888e367a3016325a56b959f0a0b7` (summary) | No |
| Matched-fault robustness | `scripts/analyze_matched_fault_support.py` via `scripts/finalize_v3_offline.py` | six promoted score files and locally obtained v3 condition manifest | `results/answer_leakage_repair/final_v3_analysis/matched_fault_support/` | 3,534 matched instances; 238 / 0 / 351 | `adfe320774547a916261456dac5a2c8dd4eb552780d342b99b2822c92c5e5052` (summary) | No |
| Public score release (row-level) | `scripts/build_pseudonymous_scores.py --input-dir results/final_main_sanitized --output-dir results/final_main_tier_b` | six merged v3 result files | **not shipped** — row-level; excluded from the aggregate-only release | 4,891 rows/provider (developer tree only) | n/a (excluded) | No |
| Row-level aggregate verification | `scripts/verify_tier_b_release.py --mapping <private>` | Tier-B scores + private mapping + frozen manifests | console `PASS` report; developer-tree only | 6 provider primaries + control-plane + matched-fault identical | n/a (verification) | No |
| Scientific figures | `scripts/build_system_validation_paper_artifacts.py` | separate frozen system-validation outputs | `results/system_validation_paper/figure_cross_system_effects.{png,pdf}` | independent experiment; not included in this clean branch | not applicable | No; no final-main figure generator found |

## JSS final-evidence additions

| Evidence | Generating code | Public aggregate | Private dependency |
|---|---|---|---|
| RQ1 Observability V2 | `scripts/analyze_jss_rq1_observability.py`, `scripts/audit_jss_rq1_observability.py` | `results/jss_rq1_observability_v2/`, `results/jss_public_aggregate_v1/rq1_observability_v2.json` | full v3 manifest and six row-level provider files |
| Scoring consistency and sensitivity | `scripts/analyze_jss_scoring_consistency.py` | `results/jss_public_aggregate_v1/scoring_consistency_sensitivity.json` | final block-level Formal-300 data, observations, private human adjudication, and semantic payloads |
| Native base-cluster bootstrap | `scripts/analyze_formal_300_subgroups.py` | `results/formal_300_subgroup_v1/bootstrap_comparison.json` | private final block-level Formal-300 input and selected-block metadata |
| JSS interpretation and limitations | `docs/JSS_PUBLIC_RESULTS.md` | identifier-free documentation | none |

The public RQ1 summary drops private artifact paths, block/observation
identifiers, and content fingerprints while retaining all aggregate estimates
and bootstrap metadata. The scoring-consistency output contains only aggregate
counts, effects, intervals, and treatment definitions.

## Frozen specifications

- `configs/main_analysis_spec_v1.json` — `a9e7abf6b14b584d6ce5e8808c3031eed6998ea6c532c32c2a7dcc98e1c01cee`
- `configs/main_failure_handling_spec_v1.json` — `a2da7d7548957bc68a68530775cad3af917a80433168d5ad778b7c03b0572e43`
- Base freeze metadata — `dc38d423242b655e729549a970b1185e5c50318a6c7ec849238098f04a87ec97`
- Condition freeze metadata — `bed355d0942087a001f8a5179ac001143c56dd747e5ce678a3bf5a06f9993ba9`

The sanitized score mapping and the row-level score files are developer-tree
artifacts only; they are **not** part of the aggregate-only release. In the
snapshot, `MANIFEST.sha256` records the hashes of the aggregate files that are
actually shipped.

## B4 scoring-validation artifacts

`results/b4_scoring_validation/` is a separate validation output directory;
it is never used to overwrite or replace `results/final_main_tier_b/`.
`scripts/validate_b4_scoring.py` compares existing model answers against
locally supplied native decisions and reports deterministic, native,
adjudication-required, and unscored outcomes. `scripts/validate_oaks_parity.py`
calculates OAKS option parity, including ambiguous/unscored rows. The scripts
make no network calls and do not rerun answer generation. Private/licensed
LongMemEval, OAKS Novel, and Memora inputs are required for actual native
agreement evidence; when absent, the script emits a blocker report rather
than fabricating counts. Memora dry-run judge plans represent three judges
per request and are payload construction only.

## Answer-leakage repair outputs

- Historical v2 audit: `results/answer_leakage_audit.json` (`FAIL`, 53 bases,
  159 rows).
- Corrected v3 audit: `results/answer_leakage_repair/answer_leakage_audit_v3.json`
  (`PASS` for the repaired manifest only).
- Exact changed-row comparison: `changed_conditions.jsonl` and
  `changed_condition_ids.txt` — **excluded** (row-level, source-derived ids).
- Official evidence mapping: `source_evidence_provenance.json` — **excluded**.
- Pending provider manifests: `provider_rerun_manifests/{openai,anthropic,deepseek,gemini,kimi,grok}.jsonl`.
- The six v3 merged inputs were validated before sanitization. Anthropic retains
  six documented final failures under the frozen failure policy; they were not
  retried or edited during finalization.
- V2 results remain historical provenance. V3 public files contain 4,626
  unchanged historical rows and 265 validated replacements per provider.
- `results/answer_leakage_repair/final_v3_analysis/old_vs_new_statistics.md`
  explicitly lists every changed summary statistic; no old value should be
  quoted without checking that comparison.
- `data/final_main_governance_condition_manifest_v3.jsonl` and
  `provider_rerun_manifests/*.jsonl` contain benchmark-derived text and are
  deliberately excluded pending licensing approval.
