# B4 CodeX Audit

Read-only audit completed. No files were modified, restored, generated, or deleted. No API/network calls were made.

## 1. LongMemEval

### Evidence found

- `data/final_main_governance_base_manifest.jsonl`
  - 191 LongMemEval bases.
  - Contains benchmark-derived questions, expected answers, source evidence, and IDs.
  - Not a native-evaluator validation artifact.

- `data/final_main_governance_base_manifest_v2.jsonl`
  - Same 191-base LongMemEval cohort.
  - Contains `expected_answer`, but no native evaluator output.

- `data/official_longmemeval_oracle_20.jsonl`
- `data/official_longmemeval_oracle_100.jsonl`
- `data/official_longmemeval_oracle_200.jsonl`
  - Native-source-style benchmark examples.
  - These are 20/100/200-example datasets, not the exact 72-sample validation cohort.

- `results/final_main_grok_43_final/scored/full_results_scored.jsonl`
  - Existing answer-generation output for one provider.
  - Contains 1,111 LongMemEval rows, model answers, oracle answers, and GovMem custom scores.
  - It is not native-evaluator output.

- `results/final_main_sanitized/{provider}.jsonl`
  - Six providers, 1,111 LongMemEval rows each.
  - Response text and oracle answers are removed.

- `data/final_main_governance_condition_v3.freeze.json`
  - References the local source path:
    `longmemeval-cleaned/longmemeval_oracle.json`
  - Records source SHA-256:
    `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`

### Historical commits

Related commits:

- `31a0cd1` — introduced the source-aware scoring pipeline and final-main manifests.
- `df9bcd5` — LongMemEval evidence-leakage repair and v3 source provenance.
- `f656132` — final public artifact release.
- `0f34190` — six-provider/Grok result freeze.

No commit contains an explicit B4 LongMemEval validation report.

### 67/72 reproducibility

Not independently reproducible from the repository.

Searches found:

- no literal `67/72`;
- no literal `67 / 72`;
- no 72-row validation cohort;
- no five-disagreement case list;
- no native evaluator output;
- no validation-specific IDs or manifest.

### Missing inputs

- Exact 72-example validation cohort.
- Native LongMemEval labels/evaluator outputs.
- The five disagreement records.
- Any validation-specific answer-generation subset mapping.

The source benchmark path is inferable, but the validation cohort and native comparison output are not.

## 2. OAKS

### Evidence found

- `scripts/audit_oaks_novel_for_main_dataset.py`
  - Audits OAKS native question transitions.
  - Requires an external OAKS JSON/JSONL input.
  - No native answer-parity evaluator is implemented there.

- `tests/test_oaks_novel_main_dataset_audit.py`
  - Tests synthetic transition/audit behavior only.

- `data/final_main_governance_base_manifest.jsonl`
- `data/final_main_governance_base_manifest_v2.jsonl`
  - 160 OAKS bases.

- `results/final_main_sanitized/{provider}.jsonl`
  - Each provider contains:
    - 1,120 OAKS condition rows;
    - 160 clean;
    - 480 corrupted;
    - 480 enforced.
  - Six providers therefore give:

    `6 × 1,120 = 6,720 rows`

- `results/final_main_grok_43_final/scored/full_results_scored.jsonl`
  - Contains one provider’s 1,120 OAKS rows.
  - Includes `task_oracle_answer`, resolved option, model answer, and custom correctness.
  - This is answer-generation/custom-scoring output, not an independent native-parity report.

### Historical commits

- `31a0cd1` introduced:
  - `eval/system_validation_scoring.py`
  - `scripts/audit_oaks_novel_for_main_dataset.py`
  - OAKS-related final-main manifest support.

- `0f34190` introduced the tracked Grok full scored output:
  - `results/final_main_grok_43_final/scored/full_results_scored.jsonl`

No commit contains a `6,720 / 6,720` native-parity artifact.

### 6,720/6,720 reproducibility

The denominator is independently explainable from public/current artifacts:

- 160 OAKS bases;
- 1,120 conditions per provider;
- six providers;
- 6,720 total provider-condition rows.

The exact native-parity claim is not independently reproducible because the repository lacks:

- the native OAKS source release;
- the exact validation input;
- independent native option labels/evaluator output;
- the original parity comparison report.

The current custom scorer can be inspected and one provider’s custom-scored rows are present, but that is not sufficient to establish independent native parity.

### Missing inputs

- OAKS source JSON/JSONL release.
- Exact native option-correctness validation cohort.
- Native option labels or benchmark-native evaluator output.
- Six-provider validation comparison inputs, if the 6,720 total refers to all providers.

## 3. Memora

### Evidence found

- `eval/memora_adapter.py`
  - Loads local Memora persona/period/data directories.
  - Expects `conversations/session_*.json` and `evaluation_questions_*.json`.

- `tests/fixtures/memora-mini/`
  - Small synthetic/local adapter fixture.
  - Not the B4 validation cohort.

- `data/system_validation_frozen_v1.jsonl`
  - Contains 60 system-validation rows total, including 60 Memora rows across the broader system-validation experiment.
  - This is not the claimed approximately 60-sample Memora B4 cohort.

- `data/governance_fault_rq1_pilot_frozen*.jsonl`
  - Contains Memora rows for separate RQ1 governance-fault experiments.
  - Not B4 native scoring validation.

- `results/final_main_grok_43_final/scored/full_results_scored.jsonl`
  - Contains 2,660 Memora answer-generation rows for one provider.
  - Subtype counts:
    - preference recommendation: 1,400
    - project proposal: 581
    - meeting notes: 245
    - email writeup: 238
    - activity todos: 196
  - Contains model answers, oracle structures, and GovMem custom scores.
  - No native/judge decisions.

- `results/final_main_sanitized/{provider}.jsonl`
  - Contains only score metadata and no Memora task/oracle text.

### FAMA/native evaluation evidence

No FAMA code, judge prompt, judge request, judge output, criterion extraction artifact, or adjudication result was found.

Search results contained no:

- `fama`;
- native Memora judge implementation;
- judge output directory;
- native criterion JSONL;
- 911-criterion report.

### Historical commits

Memora-related history contains:

- `7c7245e` — Memora dataset adapter.
- `31a0cd1` — final-main scoring and manifests.
- `0f34190` — Grok final-main scored output.
- Earlier RQ1/system-validation commits.

No historical commit contains the approximate 60-sample/911-criterion B4 artifact.

### Approximately 60 samples / 911 criteria

Not independently reproducible.

The repository has Memora samples and answer-generation outputs, but not:

- the matched B4 cohort;
- the atomic criterion extraction;
- native criterion labels;
- judge outputs;
- the original matching manifest.

Native/judge calls are still needed if the goal is comparison against native-style Memora judgments. Deterministic comparison can proceed offline only after receiving the matched cohort plus native criteria/labels. Paid judging remains unnecessary for the audit itself and is not currently implemented or invoked.

## 4. Public-release gap

### Apparently omitted artifacts

The following are explicitly ignored or documented as local/private:

- `data/final_main_governance_condition_manifest_v2.jsonl`
- `data/final_main_governance_condition_manifest_v3.jsonl`
- `results/answer_leakage_repair/provider_rerun_manifests/`
- `results/answer_leakage_repair/source_evidence_provenance.json`
- merged provider JSONL;
- rerun directories;
- logs/checkpoints;
- raw benchmark downloads;
- text-bearing manifests.

Relevant documentation:

- `.gitignore:17-28`
- `ARTIFACTS.md:5-18`
- `ARTIFACTS.md:25-28`
- `ARTIFACTS.md:53-65`
- `PUBLIC_BRANCH_REPORT.md:93-143`
- `FINALIZATION_REPORT.md:108-120`
- `results/answer_leakage_repair/final_v3_analysis/licensing_blocker_report.json`

The documented reason is licensing/privacy:

- benchmark-derived text;
- raw benchmark data;
- answer-bearing fields;
- provider payloads and request metadata;
- raw SDK responses;
- credentials and logs.

The public-release documentation does not identify a specific omitted B4 artifact by name. The exact B4 validation artifacts appear never to have been committed to Git, rather than being deleted from Git history.

### Important existing raw artifact

This tracked file remains present:

`results/final_main_grok_43_final/scored/full_results_scored.jsonl`

It contains raw model answers, oracle answers, benchmark-derived task text, request metadata, and custom scores. It is useful as an existing answer-generation artifact, but it is not native validation evidence.

## 5. Next step

Minimum files needed:

### LongMemEval

- Exact 72-row validation cohort or ID list.
- Existing model answers for those IDs.
- Native evaluator decisions or native labels.
- Five disagreement case records, preferably ID-only plus local lookup metadata.

### OAKS

- Licensed OAKS source JSON/JSONL.
- Exact validation cohort or complete six-provider row set.
- Native option labels/evaluator results.
- Existing model answers or raw scored rows sufficient to map each comparison.

### Memora

- Approximately 60 matched validation sample IDs.
- Existing model answers for those samples.
- Approximately 911 extracted atomic criteria.
- Native/human labels or FAMA-style judge outputs.
- Judge prompts/request schema if historical reproducibility is required.

### Reproducible offline now

- The 731-base source split.
- The OAKS denominator calculation:
  160 bases, 1,120 conditions/provider, 6 providers = 6,720 rows.
- Existing custom-score inspection for the tracked Grok full scored output.
- Existing deterministic scorer/unit tests.
- B4 blocker report generation.

### Still blocked

- LongMemEval 67/72 native agreement.
- OAKS 6,720/6,720 native parity.
- Memora approximately 60/911 native agreement.
- Any claim involving native evaluator outputs or FAMA judgments.

## Exact shell commands used

```bash
git status --short

rg -n -i --hidden --glob '!/.git/**' \
  --glob '!results/final_main_sanitized/*.jsonl' \
  '67[[:space:]]*/[[:space:]]*72|67[[:space:]]*/[[:space:]]*72|93\.1|6[,.]?720|\b911\b|native|parity|agreement|fama|adjudication|matched|memora|longmemeval|oaks' \
  results data scripts eval tests docs README.md ARTIFACTS.md FINALIZATION_REPORT.md PUBLIC_BRANCH_REPORT.md .gitignore

find results data scripts eval tests docs -type f -print | sort \
  | rg -i 'b4|native|valid|parity|fama|adjud|match|memora|longmemeval|oaks'

git log --all --oneline --decorate --max-count=80

git log --all --oneline -S"67/72" -- .
git log --all --oneline -S"93.1" -- .
git log --all --oneline -S"6720" -- .
git log --all --oneline -S"911" -- .
git log --all --oneline -S"native" -- scripts eval results
git log --all --oneline -S"parity" -- scripts eval results

git log --all --diff-filter=D --name-status \
  --pretty=format:'COMMIT %H %s'

git log --all --pretty=format: --name-only \
  | sort -u \
  | rg -i '(b4|fama|judge|adjud|native.*(valid|eval)|valid.*native|parity)'

git log --all --pretty=format: --name-only \
  | sort -u \
  | rg -i 'oaks|novel'

git log --all --pretty=format: --name-only \
  | sort -u \
  | rg -i 'memora'

git show --stat --oneline 31a0cd1
git show --stat --oneline f656132
git show --stat --oneline 0f34190
git show --stat --oneline df9bcd5

git show --name-only --format='' 0f34190 \
  | rg 'full_results_scored|results/final_main'

git ls-tree -r --name-only 0f34190 \
  | rg 'results/.*/full_results_scored.jsonl'

find results -type f -print \
  | rg 'full_results_scored|scored|analysis_summary|summary'

python - <<'PY'
import json
from collections import Counter
from pathlib import Path
# Read-only count inspection of manifests and score files.
PY

nl -ba .gitignore
nl -ba ARTIFACTS.md
nl -ba README.md
nl -ba PUBLIC_BRANCH_REPORT.md
nl -ba FINALIZATION_REPORT.md

cat results/b4_scoring_validation/b4_validation_summary.json
```
