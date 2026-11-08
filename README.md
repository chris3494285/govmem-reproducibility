# Stress-Testing the Control Plane of Agent Memory

This repository contains the reproducibility pipeline for the paper
“Stress-Testing the Control Plane of Agent Memory.” The study asks whether
governance faults in long-term agent memory propagate through admission and
context construction into downstream answer quality, and whether enforcement
restores the intended state.

The final cohort has 731 unique bases: 191 LongMemEval, 380 Memora, and 160
OAKS Novel. Each base has one clean sibling and one corrupted/enforced sibling
for every supported fault, giving 2,080 supported base-fault pairs and 4,891
condition rows (731 clean, 2,080 corrupted, 2,080 enforced). The four fault
operators are missing, stale, flipped, and conflicting. The frozen cross-
provider analysis represents OpenAI, Anthropic, DeepSeek, Gemini, Kimi, and
xAI/Grok.

## Release status

The public JSS reproducibility release is available at
https://github.com/chris3494285/govmem-jss-reproducibility.

The historical v2 formal condition manifest contains a documented
answer-leakage blocker. In 159 rows across 53 LongMemEval bases, the
model-visible evidence was constructed with a fallback derived directly from
`expected_answer`. V2, its audit, frozen results, and commit history are
preserved byte-for-byte.

The corrected v3 manifest reconstructs all five siblings for each affected
base from official native LongMemEval user-message evidence. All six selective
reruns are complete and merged offline.

This repository publishes an **aggregate-only** release. It ships public
experiment code, offline tests, non-secret configurations, documentation, and
aggregate tables/summaries. It ships **no row-level result file**: the
per-example score files (`results/final_main_tier_b/`,
`results/final_main_sanitized/`), the per-instance/per-base tables, the
cohort/judge-request/validation dumps, the formal-300 execution plan, and the
clean-repeat generation/judging ledgers are all withheld, together with every
source-derived benchmark identifier and per-row content fingerprint. The
resulting reproducibility tiers are spelled out in
`docs/REPRODUCTION_TIERS.md`.

Recomputed v3 summaries and the complete old-versus-new statistic comparison
are under `results/answer_leakage_repair/final_v3_analysis/`. The public
offline suite passes with **525 tests passed, 230 skipped, and 0 failed**.
Full end-to-end reproduction may still require separately obtaining permitted
upstream benchmark inputs and access to private experiment records that are
not redistributed.

Final v3 headline values are corruption −27.55 pp, recovery +27.51 pp,
context-changed corruption −26.04 pp, context-unchanged corruption −1.28 pp,
control-plane change 96.25%, and restoration 100%.

The final JSS evidence, including RQ1 Observability V2, the native
block/base-cluster bootstrap distinction, and the isolated scoring
inconsistency sensitivity, is summarized in
`docs/JSS_PUBLIC_RESULTS.md`. Aggregate-only machine-readable versions are under
`results/jss_public_aggregate_v1/`.

```bash
python scripts/repair_answer_leakage.py \
  --source "$LONGMEMEVAL_SOURCE/longmemeval-cleaned/longmemeval_oracle.json"
python scripts/audit_answer_leakage.py
```

The historical audit remains at `results/answer_leakage_audit.json`; audit it
explicitly with `python scripts/audit_answer_leakage.py --manifest
data/final_main_governance_condition_manifest_v2.jsonl --report-only`. The
default audit targets v3 and passes with zero fallback rows and zero unresolved
source provenance.

## Repository layout

- `govmem/`: governance schemas, fault operators, enforcement, and memory systems.
- `eval/`: benchmark adapters, scoring, provider implementations, batch lifecycle,
  token budgets, and offline analysis support.
- `scripts/`: frozen manifest builders, inference runners, audits, analyses, and
  deterministic sanitization.
- `data/`: frozen-manifest freeze metadata and system-validation metadata hashes.
  Raw benchmark manifests and source text are not included; benchmark text is
  not a substitute for the original licensed downloads.
- `configs/`: frozen analysis/failure specifications and provider budget configs.
- `results/`: frozen **aggregate** summaries, tables, and diagnostics. Row-level
  score files are excluded from this release.
- `ARTIFACTS.md`: claim-to-script-to-file mapping and hashes.

## Zero-API reproduction

All commands below use persisted local artifacts and make no model API calls.
They require Python 3.10+, the package dependencies, and the optional
`repro` extra. Typical runtime is seconds for audits and minutes for the full
pytest suite; sanitization of the six 4,891-row files usually takes under a
minute.

```bash
python -m pip install -e '.[repro]'
python -m pytest -q
python scripts/audit_answer_leakage.py --report-only
python scripts/audit_governance_evidence.py
python scripts/analyze_control_plane_propagation.py
python scripts/analyze_matched_fault_support.py
python scripts/finalize_v3_offline.py
python scripts/sanitize_final_results.py
```

The repair comparison, completed rerun provenance, and offline v3 analyses are
in `results/answer_leakage_repair/`. Raw merged/rerun JSONL and text-bearing
manifests are local-only and are not part of the public release. Users must
obtain LongMemEval, Memora, and OAKS Novel from their official sources and
rebuild any required manifest locally. No provider API is used by any command
in this section.

The v3 cross-provider summaries and RQ-style tables are generated from the six
promoted score files by `scripts/finalize_v3_offline.py`; its single-provider
mode does not fabricate an independent replication run. Their generating inputs
and hashes are listed in `ARTIFACTS.md`; rerunning those analyses is offline and
does not call a provider.

## Aggregate-only score policy

This release contains **no condition-level score files**. The Tier-B
pseudonymous files (`results/final_main_tier_b/`) and the earlier
`results/final_main_sanitized/` files are both withheld: Tier-B removed
per-example identifiers and per-row fingerprints but still retained a
row/condition structure derived from the licensed benchmarks, outside the
approved aggregate-only redistribution scope. Instead of shipping those rows,
the release ships only the aggregate tables and summaries they feed.

What this preserves:

- all published aggregate statistics for final v3, B4, formal-300, and the
  clean-repeat control, computed from the frozen outputs and shipped as
  aggregate JSON/CSV;
- the shipped analysis code and specs, so the aggregates can be recomputed by a
  reader who independently obtains the licensed datasets and regenerates the
  row-level outputs.

**Reproducibility cost.** Aggregate statistics are verifiable from this release
(Tier 1). Recomputing them end to end, and any per-example analysis, require the
licensed datasets and the original local experiment outputs, or are not
possible from this release at all (Tiers 2 and 3). The full mapping is in
`docs/REPRODUCTION_TIERS.md`.

## B4 scoring validity

The main score uses GovMem's custom deterministic, source-aware scoring layer;
it does not claim to invoke every benchmark's original evaluator. The layer
records deterministic scoring separately from native-validation scoring,
adjudication-required cases, and unscored cases. B4 validation measures
agreement with benchmark-native or native-style decisions supplied from local
LongMemEval, OAKS Novel, or Memora inputs. “Original benchmark questions or
reference answers” is not a claim that the original benchmark evaluator was
used.

B4 is offline and re-scores existing answers; it does not rerun answer
generation. Licensed/private benchmark inputs are intentionally not committed.
Without them the command writes an explicit blocker report in the dedicated
`results/b4_scoring_validation/` directory. Native Memora judge requests can be
written with `--dry-run-judge`; no judge/API call is made by default.

```bash
python scripts/validate_b4_scoring.py --output-dir results/b4_scoring_validation
python scripts/validate_b4_scoring.py --input /path/to/private/b4_validation.jsonl --source longmemeval --output-dir results/b4_scoring_validation
python scripts/validate_oaks_parity.py --input /path/to/private/oaks_validation.jsonl --output results/b4_scoring_validation/oaks_validation.json
python scripts/validate_b4_scoring.py --input /path/to/private/memora_validation.jsonl --source memora --dry-run-judge --output-dir results/b4_scoring_validation
```

The local validation input must carry native decisions (`native_correctness`
or `native_score`) and must be constructed under the relevant benchmark
license. Reports contain IDs, hashes, and scoring metadata rather than
benchmark text.

To verify hashes:

```bash
shasum -a 256 data/final_main_governance_base_manifest_v2.jsonl \
  data/final_main_governance_condition_manifest_v2.freeze.json \
  data/final_main_governance_condition_manifest_v3.freeze.json
cat results/final_main_cross_provider_summary_v3/cross_provider_summary.md
cat results/answer_leakage_repair/final_v3_analysis/old_vs_new_statistics.md
( cd /path/to/release-snapshot && shasum -a 256 -c MANIFEST.sha256 )
```

## Optional new inference

New inference is outside the zero-API reproduction path and requires the
corresponding provider credential. Set only the environment variable for the
provider you intend to use; never place values in tracked files:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export DEEPSEEK_API_KEY=...
export GEMINI_API_KEY=...
export KIMI_API_KEY=...
export XAI_API_KEY=...
```

The runners are `scripts/run_final_main_openai.py`,
`run_final_main_anthropic.py`, `run_final_main_deepseek.py`,
`run_final_main_gemini.py`, `run_final_main_kimi.py`, and the xAI/Grok runner
used by the frozen artifact. Their dry-run/preflight modes do not call an API;
live modes do. Do not use live modes to reproduce frozen scores.

If a future rerun is authorized, each provider can rerun only changed
conditions with its corresponding locally built file under
`results/answer_leakage_repair/provider_rerun_manifests/`, for example:

```bash
python scripts/run_final_main_openai.py --manifest results/answer_leakage_repair/provider_rerun_manifests/openai.jsonl --results-dir results/answer_leakage_repair/rerun_openai
python scripts/run_final_main_anthropic.py --manifest results/answer_leakage_repair/provider_rerun_manifests/anthropic.jsonl --results-dir results/answer_leakage_repair/rerun_anthropic
python scripts/run_final_main_deepseek.py --manifest results/answer_leakage_repair/provider_rerun_manifests/deepseek.jsonl --results-dir results/answer_leakage_repair/rerun_deepseek
python scripts/run_final_main_gemini.py --manifest results/answer_leakage_repair/provider_rerun_manifests/gemini.jsonl --results-dir results/answer_leakage_repair/rerun_gemini
python scripts/run_final_main_kimi.py --manifest results/answer_leakage_repair/provider_rerun_manifests/kimi.jsonl --results-dir results/answer_leakage_repair/rerun_kimi
python scripts/run_final_main_grok.py --manifest results/answer_leakage_repair/provider_rerun_manifests/grok.jsonl --results-dir results/answer_leakage_repair/rerun_grok
```

These six live commands require the corresponding provider environment
variable and make API calls; they are intentionally not run here.

The private-input JSS analyses can be regenerated locally when all frozen and
private inputs are present:

```bash
python scripts/analyze_jss_scoring_consistency.py --force
python scripts/build_jss_public_aggregate.py --force
```

The first command computes the scoring-consistency sensitivity; the second
validates manuscript counts and writes the identifier-free public summary.
The public suite skips private-input integration tests by default.

## Data and licensing

The three upstream benchmark licenses have been verified against the exact
versions GovMem used and are documented, with their notices and citations, in
`THIRD_PARTY_DATASETS.md`:

| Dataset | License |
|---|---|
| LongMemEval (cleaned) | MIT |
| Memora | Apache License 2.0 |
| OAKS Novel | Adobe Research License (**noncommercial research only**) |

This remains an **aggregate-only** release: no benchmark question, answer,
context text, per-example identifier, per-row fingerprint, or text-bearing
manifest is redistributed. The original benchmark files must still be obtained
under their own terms; use the frozen hashes, source identifiers, and audit
scripts to rebuild manifests from locally acquired copies.

Because the aggregate tables depend on OAKS Novel, whose license permits
redistribution of derivatives **only for noncommercial research purposes**, this
release as a whole carries that restriction. See `THIRD_PARTY_DATASETS.md` for
the license texts, attribution, and the exact OAKS-derived artifact inventory in
`OAKS_DERIVED_ARTIFACTS.md`.

`LICENSE` contains the MIT grant for original GovMem implementation and test
code only. `LICENSE_SCOPE.md` identifies the covered and excluded material;
the MIT text does not cover third-party benchmarks, benchmark-derived
aggregate results, or upstream notices.

Coauthor public-release approval and MIT intent are recorded in
`LICENSE_APPROVAL.md`. Original documentation, configuration, and aggregate
results are distributed under `MATERIALS_REDISTRIBUTION_NOTICE.md`, not MIT.
The responsible humans have also confirmed independent personal ownership and
the absence of employer, university, funder, or contractor restrictions on the
original materials.

`CITATION.cff` records the manuscript title and author order without claiming a
DOI or journal publication date.

## Paper mapping

`ARTIFACTS.md` maps the 731/2,080/4,891 cohort claims, six-provider summaries,
RQ1–RQ4 analyses, enforcement recovery, governance evidence audit, control-
plane diagnostic, matched-fault robustness, JSS Observability V2, bootstrap
provenance, scoring-consistency sensitivity, tables, and figures to their
inputs, outputs, hashes, and API requirements.
