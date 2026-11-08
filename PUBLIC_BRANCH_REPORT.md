# GovMem Public Branch Report

> **Historical record.** This report was written for the earlier public branch,
> which shipped the row-level `results/final_main_sanitized/` score files. The
> current release is **aggregate-only**: no row-level score file is shipped.
> See `docs/REPRODUCTION_TIERS.md`.

## Base and scope

This worktree is branch `paper-v3-public` at base commit
`0f34190c5795503e444ab56f7ec5737a24c6f6b0`, exactly the starting point of
`origin/paper-v2`. The six local `paper-v2` commits and its worktree/index were
not altered. No provider API, inference, rerun, commit, push, reset, clean, or
delete was performed.

## Public-safe artifacts proposed for staging

The exact staged list is printed by `git diff --cached --name-status` after this
report is staged. It is limited to release documentation, package/source code,
tests, stable configurations, sanitized score files, hash mapping, text-free
repair summaries, and aggregate v3 analysis outputs. The six files under
`results/final_main_sanitized/` are the canonical public score artifacts; each
has 4,891 rows and 4,891 unique condition IDs.

Exact staged paths (with the aggregate directory expanded by Git) are:

```text
.gitignore
ARTIFACTS.md
PUBLIC_BRANCH_REPORT.md
README.md
pyproject.toml
data/final_main_governance_condition_manifest_v3.freeze.json
configs/final_main_token_budget_gemini_flash_v1.json
configs/final_main_token_budget_kimi_v4.json
configs/final_main_token_budget_v2.json
configs/final_main_token_budget_grok_v1.json
eval/final_main_batch.py
eval/final_main_batch_lifecycle.py
eval/final_main_rerun_identity.py
eval/final_main_token_budget.py
eval/gemini_validation.py
eval/reconstruction_provider.py
eval/providers/__init__.py
eval/providers/anthropic_provider.py
eval/providers/gemini_provider.py
eval/providers/kimi_provider.py
eval/providers/openai_provider.py
eval/providers/qwen_provider.py
scripts/analyze_control_plane_propagation.py
scripts/analyze_final_main_results.py
scripts/analyze_matched_fault_support.py
scripts/audit_answer_leakage.py
scripts/audit_final_main_transport.py
scripts/audit_governance_evidence.py
scripts/audit_rerun_identity.py
scripts/build_rq1_paper_table.py
scripts/finalize_v3_offline.py
scripts/merge_final_main_results.py
scripts/repair_answer_leakage.py
scripts/run_final_main_anthropic.py
scripts/run_final_main_deepseek.py
scripts/run_final_main_gemini.py
scripts/run_final_main_grok.py
scripts/run_final_main_kimi.py
scripts/run_final_main_openai.py
scripts/sanitize_final_results.py
results/final_main_sanitized/anthropic.jsonl
results/final_main_sanitized/deepseek.jsonl
results/final_main_sanitized/gemini.jsonl
results/final_main_sanitized/grok.jsonl
results/final_main_sanitized/kimi.jsonl
results/final_main_sanitized/openai.jsonl
results/final_main_sanitized/sha256_mapping.json
results/answer_leakage_repair/answer_leakage_audit_v3.json
results/answer_leakage_repair/changed_condition_ids.txt
results/answer_leakage_repair/changed_conditions.jsonl
results/answer_leakage_repair/final_live_commands.md
results/answer_leakage_repair/repair_summary.json
results/answer_leakage_repair/rerun_cost_estimate.json
results/answer_leakage_repair/rerun_execution_plan.md
results/answer_leakage_repair/rerun_identity_audit.json
results/answer_leakage_repair/final_v3_analysis/ (all listed aggregate files)
tests/test_answer_leakage.py
tests/test_control_plane_propagation.py
tests/test_final_main_anthropic_runner.py
tests/test_final_main_batch_cost.py
tests/test_final_main_batch_lifecycle.py
tests/test_final_main_kimi.py
tests/test_final_main_token_budget.py
tests/test_gemini_provider.py
tests/test_governance_evidence_audit.py
tests/test_matched_fault_support.py
tests/test_rerun_identity.py
tests/test_sanitize_final_results.py
```

## Excluded material

The v3 text-bearing condition manifest, provider rerun manifests, source
evidence provenance with benchmark locators, merged provider JSONL, rerun
directories, checkpoints, logs, canaries, smoke/preflight/validation/proposal/
repair/exploratory outputs, raw SDK responses, request metadata, credentials,
`.env` files, caches, virtual environments, and machine-specific files are not
proposed for staging. The base branch also contains historical text-bearing or
raw experiment paths; because the user prohibited destructive deletion, those
existing paths remain a public-branch blocker and are not staged as additions.

## Hashes and counts

| Provider | Rows | SHA-256 |
|---|---:|---|
| anthropic | 4,891 | `4e416bbadfb90b4b8c82e1d2b6acc7b9a46e86eaf6f4144fe0c15b79f10f8abf` |
| deepseek | 4,891 | `9d93ea8f1e33ff879c483463fdcc16df3578fa3d68cf4576e5f8282f47d3d4d3` |
| gemini | 4,891 | `8d237c86ee97f592ca8a4fb12fb153c3bf7afd0ca232ee3201ad3ae13e1e8da8` |
| grok | 4,891 | `9f7ac01b77e2788d6ecea888fac1b59d56a16a45fe29f6cb300c70cddda9d336` |
| kimi | 4,891 | `5a48c0e36b661432a62a1fb84ec0cad85a08c467b1cb6adfd8373346e4e8dcd3` |
| openai | 4,891 | `c803ff46b4a606eecc8d100222ca35f3743bc70446aff90558a9b2b95eac8adb` |

The mapping file records these public hashes and the private-source hashes;
its SHA-256 is `cb849f43ae476d633e48ada4db9d35f1b8f111321a67e185dd3bf105ef755988`.

## Offline checks

`compileall` passed. `audit_rerun_identity.py` passed for all six providers.
The default leakage audit and v3 finalizer could not run self-contained because
their required text-bearing v2/v3 manifests are intentionally absent. The full
pytest collection is not clean on this branch: it reports missing historical
modules omitted from the narrow public selection, including Qwen provider,
canary, and recovery helpers. Therefore the branch is not yet safe to commit
or push publicly. No external API was called.

## Licensing and scan result

No secret values are reported. The intended public score files contain no raw
prompt, response, request ID, credential, endpoint/account metadata, oracle
answer, task question, or benchmark context fields. Code/tests contain field
names and an intentional test-only secret-pattern fixture; that fixture is not
in the proposed public allowlist. Text-bearing benchmark manifests and source
provenance are excluded pending confirmation of LongMemEval, Memora, and OAKS
Novel redistribution rights. Users must obtain those datasets from official
sources.

## Comparison and readiness

The proposed branch has no commits beyond its origin base. It is a clean
worktree-derived public selection, but it is not safe to commit or push until
the existing base-branch historical payloads are removed or otherwise excluded
non-destructively, the missing test dependencies are resolved without adding
canary/exploratory artifacts, and the manifest-dependent audits are made
self-contained through approved locator/hash-only inputs or documented local
dataset acquisition.
