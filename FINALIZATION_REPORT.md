# GovMem v3 Finalization Report

> **Historical record.** The promoted row-level `results/final_main_sanitized/`
> files named below are not part of the current aggregate-only release; only
> the aggregate summaries derived from them are shipped. See
> `docs/REPRODUCTION_TIERS.md`.

## Zero API Call Confirmation

This finalization step made zero model-provider API calls, ran no inference,
and did not rerun any provider. No files were staged, committed, pushed,
deleted, reset, or cleaned.

## Promoted Public Result Hashes and Row Counts

Each promoted file contains 4,891 rows and 4,891 unique v3 manifest IDs.

| Provider | File | SHA-256 | Rows |
|---|---|---|---:|
| Anthropic | `results/final_main_sanitized/anthropic.jsonl` | `4e416bbadfb90b4b8c82e1d2b6acc7b9a46e86f6f4144fe0c15b79f10f8abf` | 4891 |
| DeepSeek | `results/final_main_sanitized/deepseek.jsonl` | `9d93ea8f1e33ff879c483463fdcc16df3578fa3d68cf4576e5f8282f47d3d4d3` | 4891 |
| Gemini | `results/final_main_sanitized/gemini.jsonl` | `8d237c86ee97f592ca8a4fb12fb153c3bf7afd0ca232ee3201ad3ae13e1e8da8` | 4891 |
| Grok/xAI | `results/final_main_sanitized/grok.jsonl` | `9f7ac01b77e2788d6ecea888fac1b59d56a16a45fe29f6cb300c70cddda9d336` | 4891 |
| Kimi | `results/final_main_sanitized/kimi.jsonl` | `5a48c0e36b661432a62a1fb84ec0cad85a08c467b1cb6adfd8373346e4e8dcd3` | 4891 |
| OpenAI | `results/final_main_sanitized/openai.jsonl` | `c803ff46b4a606eecc8d100222ca35f3743bc70446aff90558a9b2b95eac8adb` | 4891 |

The mapping file is `results/final_main_sanitized/sha256_mapping.json`, SHA-256
`cb849f43ae476d633e48ada4db9d35f1b8f111321a67e185dd3bf105ef755988`.

## Sanitization Audit

The canonical sanitizer was run from the six merged inputs into the promoted
public paths. Scientific score, condition, model, scoring, manifest, prompt,
failure, and accounting fields were preserved. Provider payloads, response
bodies, request IDs, credentials, endpoint/account metadata, prompts, and
oracle-answer fields were excluded from the promoted score files.

The six merged source hashes matched their supplied hashes before and after
sanitization. The source provenance report has 53 entries, no absolute paths,
and zero `oracle_answer_substitution` values.

## Changed Paper Values

The complete old-versus-v3 comparison is in:

- `results/answer_leakage_repair/final_v3_analysis/old_vs_new_statistics.json`
- `results/answer_leakage_repair/final_v3_analysis/old_vs_new_statistics.csv`
- `results/answer_leakage_repair/final_v3_analysis/old_vs_new_statistics.md`
- `results/answer_leakage_repair/final_v3_analysis/paper_update_checklist.md`

The comparison contains 282 statistic rows, including denominators, confidence
intervals, provider results, and changed flags; 126 provider statistics changed
in the earlier finalized comparison, with 21 changed records per provider after
denominator rows were included in the checklist.

Headline values are:

- Event-macro corruption: −31.43 pp → −27.55 pp.
- Event-macro recovery: +31.55 pp → +27.51 pp.
- Context-changed corruption: −28.01 pp → −26.04 pp.
- Unchanged-context corruption: −1.28 pp → −1.28 pp.
- Control-plane change: 96.25% → 96.25%.
- Restoration: 100% → 100%.
- Matched instances: 3,534; stale/conflicting: 238; stale/flipped: 0;
  stale/missing: 351.

The repository contains no Prism, LaTeX, manuscript, or other paper-source file,
so textual paper updates have not been performed here.

## RQ1–RQ4 Dependency and Regeneration Decisions

The final-main v3 provider summaries, source/event/fault breakdowns,
enforcement recovery, control-plane propagation, and matched-fault analyses use
the promoted v3 files.

The existing RQ1, RQ2, RQ3, and RQ4 scripts use separate frozen pilot
manifests/results. They were run offline, but they are independent experiments,
not regenerated from the repaired 4,891-row final-main files. Their outputs must
not be described as v3 final-main recomputations.

## Regenerated Tables and Figures

Generated final-main v3 outputs are under:

`results/answer_leakage_repair/final_v3_analysis/`

This includes provider summaries, cross-provider summary, propagation,
matched-fault support, governance evidence, old-versus-new comparisons, and
the checklist.

The separate system-validation tables and figure were regenerated under:

- `results/system_validation_analysis/`
- `results/system_validation_statistics/`
- `results/system_validation_paper/figure_cross_system_effects.png`
- `results/system_validation_paper/figure_cross_system_effects.pdf`

RQ1–RQ4 pilot outputs were regenerated offline under `results/rq1_paper/`,
`results/rq2/`, `results/rq3/`, and `results/rq4/`. No canonical generator for
additional final-main paper figures was found, and no manuscript figure source
exists in this repository. Any such figures remain a blocker.

## Licensing and Redistribution Blockers

The promoted sanitized score files contain no detected raw benchmark text.
However, proposed manifest artifacts expose benchmark-derived content:

- `data/final_main_governance_condition_manifest_v3.jsonl` exposes
  `model_visible_context_text` in 2,893 rows, other long benchmark-derived text
  fields in 2,370 occurrences, plus task questions and answer-bearing fields.
- Each file under
  `results/answer_leakage_repair/provider_rerun_manifests/` exposes
  `model_visible_context_text` in 231 rows and additional task fields.
- `results/answer_leakage_repair/source_evidence_provenance.json` exposes
  source filenames, session/message locators, example IDs, and content hashes,
  but no message content.

Redistribution terms for LongMemEval, Memora, and OAKS Novel are not established
in this repository. These manifest/provenance artifacts require license review,
or replacement with locator/hash-only artifacts and deterministic acquisition
instructions, before public release.

## Test and Audit Results

The following completed successfully offline:

- `python scripts/finalize_v3_offline.py`
- `python scripts/audit_answer_leakage.py` — PASS, 0 affected rows.
- `python scripts/audit_rerun_identity.py` — all six providers PASS.
- `python -m compileall -q govmem eval baselines scripts`
- `python -m pytest -q` — PASS, zero failures and zero collection errors.
- `git diff --check` — PASS.

The governance evidence audit passed with 731 fully supported bases and 0
unsupported bases. Anthropic's six repair-slice failures remain present under
the frozen failure policy and were not retried or edited.

## Exact Git Add Allowlist

This is a proposed allowlist only; nothing has been staged.

Tracked files:

- `README.md`
- `ARTIFACTS.md`
- `eval/final_main_rerun_identity.py`
- `eval/final_main_result_join.py`
- `scripts/sanitize_final_results.py`
- `scripts/analyze_final_main_results.py`
- `scripts/analyze_control_plane_propagation.py`
- `scripts/analyze_matched_fault_support.py`
- `scripts/audit_governance_evidence.py`
- `scripts/run_final_main_anthropic.py`
- `scripts/run_final_main_deepseek.py`
- `scripts/run_final_main_gemini.py`
- `scripts/run_final_main_grok.py`
- `scripts/run_final_main_kimi.py`
- `scripts/run_final_main_openai.py`
- `scripts/finalize_v3_offline.py`
- `data/final_main_governance_condition_manifest_v3.jsonl`
- `data/final_main_governance_condition_manifest_v3.freeze.json`
- `results/final_main_sanitized/anthropic.jsonl`
- `results/final_main_sanitized/deepseek.jsonl`
- `results/final_main_sanitized/gemini.jsonl`
- `results/final_main_sanitized/grok.jsonl`
- `results/final_main_sanitized/kimi.jsonl`
- `results/final_main_sanitized/openai.jsonl`
- `results/final_main_sanitized/sha256_mapping.json`
- `FINALIZATION_REPORT.md`

Untracked artifacts proposed only after licensing review:

- `results/answer_leakage_repair/answer_leakage_audit_v3.json`
- `results/answer_leakage_repair/changed_condition_ids.txt`
- `results/answer_leakage_repair/changed_conditions.jsonl`
- `results/answer_leakage_repair/repair_summary.json`
- `results/answer_leakage_repair/source_evidence_provenance.json`
- `results/answer_leakage_repair/final_live_commands.md`
- `results/answer_leakage_repair/rerun_cost_estimate.json`
- `results/answer_leakage_repair/rerun_execution_plan.md`
- `results/answer_leakage_repair/rerun_identity_audit.json`
- `results/answer_leakage_repair/provider_rerun_manifests/anthropic.jsonl`
- `results/answer_leakage_repair/provider_rerun_manifests/deepseek.jsonl`
- `results/answer_leakage_repair/provider_rerun_manifests/gemini.jsonl`
- `results/answer_leakage_repair/provider_rerun_manifests/grok.jsonl`
- `results/answer_leakage_repair/provider_rerun_manifests/kimi.jsonl`
- `results/answer_leakage_repair/provider_rerun_manifests/openai.jsonl`
- `results/answer_leakage_repair/final_v3_analysis/`

## Exact Do-Not-Commit List

- `logs/`
- `results/answer_leakage_repair/merged_anthropic.jsonl`
- `results/answer_leakage_repair/merged_deepseek.jsonl`
- `results/answer_leakage_repair/merged_gemini.jsonl`
- `results/answer_leakage_repair/merged_grok.jsonl`
- `results/answer_leakage_repair/merged_kimi.jsonl`
- `results/answer_leakage_repair/merged_openai.jsonl`
- `results/answer_leakage_repair/rerun_anthropic/`
- `results/answer_leakage_repair/rerun_deepseek/`
- `results/answer_leakage_repair/rerun_gemini/`
- `results/answer_leakage_repair/rerun_grok/`
- `results/answer_leakage_repair/rerun_kimi/`
- `results/answer_leakage_repair/rerun_openai/`
- `results/answer_leakage_repair/anthropic_checkpoints/`
- `results/concurrency_validation_w1/`
- `results/concurrency_validation_w4/`
- `results/concurrency_validation_w8/`
- `configs/final_main_token_budget_anthropic_repair_v1.json`
- `configs/final_main_token_budget_gemini_flash_repair_v1.json`
- `configs/final_main_token_budget_gemini_v1.json`
- `configs/final_main_token_budget_kimi_v1.json`
- `configs/final_main_token_budget_kimi_v2.json`
- `configs/final_main_token_budget_kimi_v3.json`
- all untracked `results/final_main_*` canary, smoke, preflight, proposal,
  validation, repair, failed-ID, and exploratory directories
- all untracked historical experiment outputs under `results/official_*`,
  `results/longmemeval_*`, and unrelated demo directories
- `.env`, credentials, API keys, request metadata, raw SDK responses, caches,
  virtual environments, and machine-specific files

## Remaining Submission-Readiness Blockers

`submission_ready=true` must not be asserted yet. Remaining blockers are:

1. Licensing approval for benchmark-derived text, manifest fields, locators,
   and fingerprints, or creation of approved locator/hash-only replacements.
2. Author-side update of the external manuscript and any absent final-main
   figures; no Prism/LaTeX/manuscript source is present here.
3. Author review and acceptance of all corrected paper values in the v3
   comparison.
4. Final human review of the exact staging allowlist; this task intentionally
   did not stage or commit anything.
