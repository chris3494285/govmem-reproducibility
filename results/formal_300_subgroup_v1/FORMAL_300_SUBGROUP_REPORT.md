# Formal-300 Subgroup Analysis (final human-adjudicated artifacts)

Offline analysis of the frozen Formal-300 validation. No LLM or API call, no new scoring, no response generation, and no change to any frozen result or evaluation decision. All values are recomputed from `formal_300_block_level_results.csv` (final human-adjudicated analysis; protocol `formal-300-validation-v1`, evaluator `final-native-evaluation-v2`, frozen block fingerprint `<FORMAL_BLOCK_FINGERPRINT>`).

## 1. Reconciliation with the frozen reference values

| Frozen reference | Reference value | Recomputed | Status |
|---|---:|---:|---|
| Selected blocks | 300 | 300 | match |
| Resolved blocks | 298 | 298 | match |
| Resolved observations | 898 | 898 | match |
| Observations inside complete triplets (analysis input) | not specified | 894 | derived |
| Distinct bases (300 selected blocks) | 239 | 239 | match |
| Distinct bases (298 resolved blocks) | not specified | 238 | definitional difference |
| Corruption effect | +35.54 pp | +35.54 pp | match |
| Recovery effect | +32.67 pp | +32.67 pp | match |

### Discrepancy reported, not adjusted

The reference value **239 distinct bases** is a property of the **300 selected blocks**. The **298 resolved blocks** contain **238 distinct bases**: the two unresolved blocks remove one base entirely and one base is retained through its other provider blocks. Both numbers are correct under their own definition. Every statistic below uses the 298 resolved blocks; no value was changed to match either figure.

A second counting nuance is reported for the same reason: the frozen metric counts **898 resolved observations** (900 − 2 unresolved), while the triplet analysis consumes **894 observations** (298 blocks × 3), because a block with any unresolved component is excluded entirely rather than partially used. The two numbers are not in conflict; they count different things.

The two excluded blocks are described by stratum only (no identifiers): longmemeval / deepseek / conflicting; longmemeval / gemini / conflicting. Both are LongMemEval conflicting-fault blocks whose primary judgment returned `adjudication_required`; neither was imputed.

## 2. Outcome units (deliberately not pooled into one accuracy)

* **LongMemEval** — native-style *binary* answer correctness (deterministic native / abstention routing plus blinded semantic judgment with binary projection).
* **Memora** — native *continuous* FAMA per answer, using the benchmark's own criterion-level aggregation `FAMA = max(0, MPA − λ(1 − FAA))`.
* The pooled row is an equal-weight block-average effect size across both datasets. It is **not** a single benchmark-native accuracy and must not be reported as one.

## 3. Dataset-specific results

| Dataset | Valid triplets | Base clusters | Clean | Corrupted | Enforced | Corruption (pp) | Recovery (pp) |
|---|---:|---:|---:|---:|---:|---|---|
| LongMemEval | 148 | 105 | 0.709 | 0.304 | 0.696 | $+40.54$ [+31.76, +49.32] | $+39.19$ [+30.41, +47.97] |
| Memora | 150 | 133 | 0.671 | 0.365 | 0.628 | $+30.60$ [+23.25, +37.98] | $+26.24$ [+19.05, +33.53] |
| **Pooled** | **298** | 238 | 0.690 | 0.335 | 0.662 | **$+35.54$** [+29.78, +41.12] | **$+32.67$** [+26.94, +38.36] |

Intervals are the frozen block bootstrap. The base-cluster bootstrap gives the same
picture (§5). Both datasets show the corruption effect and the recovery effect in the
same direction.

## 4. Provider-specific results

| Provider | Valid triplets | LME / Memora | Clean | Corrupted | Corruption (pp) [block CI] | Recovery (pp) [block CI] | Stability |
|---|---:|---|---:|---:|---|---|---|
| openai | 50 | 25 / 25 | 0.862 | 0.351 | $+51.13$ [+37.91, +64.18] | $+50.03$ [+36.41, +63.62] | estimated |
| anthropic | 50 | 25 / 25 | 0.822 | 0.316 | $+50.60$ [+36.55, +64.30] | $+40.38$ [+26.28, +54.70] | estimated |
| deepseek | 49 | 24 / 25 | 0.581 | 0.406 | $+17.46$ [+5.47, +29.93] | $+19.94$ [+7.77, +32.60] | estimated |
| gemini | 49 | 24 / 25 | 0.813 | 0.372 | $+44.04$ [+29.36, +58.10] | $+41.52$ [+25.81, +56.25] | estimated |
| kimi | 50 | 25 / 25 | 0.645 | 0.330 | $+31.53$ [+19.01, +44.38] | $+26.11$ [+13.77, +39.11] | estimated |
| xai | 50 | 25 / 25 | 0.419 | 0.236 | $+18.27$ [+3.78, +32.15] | $+17.97$ [+5.06, +30.62] | estimated |

### provider × dataset cells

| Provider | Dataset | Valid triplets | Corruption (pp) [block CI] | Recovery (pp) [block CI] | Stability |
|---|---|---:|---|---|---|
| openai | LongMemEval | 25 | $+52.00$ [+32.00, +72.00] | $+52.00$ [+32.00, +72.00] | degenerate:restoration_gap |
| openai | Memora | 25 | $+50.26$ [+32.57, +67.11] | $+48.07$ [+29.19, +66.07] | estimated |
| anthropic | LongMemEval | 25 | $+44.00$ [+20.00, +64.00] | $+40.00$ [+16.00, +60.00] | estimated |
| anthropic | Memora | 25 | $+57.21$ [+41.98, +72.50] | $+40.76$ [+23.46, +58.03] | estimated |
| deepseek | LongMemEval | 24 | $+33.33$ [+16.67, +54.17] | $+37.50$ [+20.73, +58.33] | estimated |
| deepseek | Memora | 25 | $+2.22$ [-10.73, +15.12] | $+3.09$ [-9.23, +15.84] | estimated |
| gemini | LongMemEval | 24 | $+54.17$ [+29.17, +75.00] | $+50.00$ [+25.00, +75.00] | estimated |
| gemini | Memora | 25 | $+34.31$ [+18.00, +51.03] | $+33.39$ [+17.54, +49.78] | estimated |
| kimi | LongMemEval | 25 | $+36.00$ [+16.00, +56.00] | $+32.00$ [+16.00, +52.00] | estimated |
| kimi | Memora | 25 | $+27.06$ [+10.33, +44.37] | $+20.22$ [+3.59, +37.82] | estimated |
| xai | LongMemEval | 25 | $+24.00$ [+0.00, +48.00] | $+24.00$ [+4.00, +44.00] | estimated |
| xai | Memora | 25 | $+12.54$ [-3.89, +29.14] | $+11.94$ [-3.89, +27.47] | estimated |

Each provider × dataset cell holds exactly 24–25 blocks and 24–25 base clusters: the frozen sampler admits each base at most once per provider, so base reuse happens only *across* providers, never inside a cell. These are **small cells with unstable intervals**; they are reported for completeness and must not be used to rank providers or to assert provider differences.

## 5. Dependence-aware uncertainty (base-cluster bootstrap)

Predeclared design:

* **Sampling unit:** one matched (generating provider, base) block carrying clean, corrupted, and enforced answers; the triplet is never split.
* **Cluster identifier:** the base example. Blocks that share a base (the same content reused across providers) are resampled together.
* **Stratification:** clusters are resampled independently within each dataset stratum (LongMemEval 148 blocks, Memora 150 blocks), preserving the dataset split. Dataset-specific, provider-specific, and provider × dataset cells are single-stratum resamples over the clusters present in that cell.
* **Iterations:** 10,000 (bootstrap_comparison.json records the same).
* **Seed:** 20261010 — the same constant as the frozen block bootstrap, so the comparison isolates the resampling scheme rather than the random stream.
* **Aggregation and weighting:** equal weight per block; a resampled cluster contributes all of its blocks, so a base drawn $m$ times contributes $m$ copies of each of its blocks. No reweighting and no down-weighting of reused bases.
* **Missing / unresolved handling:** the two blocks with an unresolved component are excluded before resampling; nothing is imputed and no partial triplet is used.

| Subset | Blocks | Clusters | Effect | Block bootstrap (frozen) | Base-cluster bootstrap |
|---|---:|---:|---|---|---|
| Pooled | 298 | 238 | corruption | [+29.78, +41.12] | [+29.40, +41.90] |
| Pooled | 298 | 238 | recovery | [+26.94, +38.36] | [+26.42, +39.09] |
| LongMemEval | 148 | 105 | corruption | [+31.76, +49.32] | [+30.32, +50.71] |
| LongMemEval | 148 | 105 | recovery | [+30.41, +47.97] | [+28.85, +49.66] |
| Memora | 150 | 133 | corruption | [+23.25, +37.98] | [+23.31, +37.69] |
| Memora | 150 | 133 | recovery | [+19.05, +33.53] | [+19.09, +33.34] |

The frozen block-bootstrap intervals are **reproduced exactly and not replaced**: frozen corruption [+29.78, +41.12], reproduced [+29.78, +41.12]; frozen recovery [+26.94, +38.36], reproduced [+26.94, +38.36].

### Comparison of the two schemes

* **Corruption (pooled).** Block CI [+29.78, +41.12]; base-cluster CI [+29.40, +41.90]; width changes from 11.33 pp to 12.49 pp.
* **Recovery (pooled).** Block CI [+26.94, +38.36]; base-cluster CI [+26.42, +39.09]; width changes from 11.43 pp to 12.67 pp.

The two schemes agree on the direction and on the qualitative conclusion for this frozen sample. They are not interchangeable: the block bootstrap treats 298 blocks as independent, while the cluster bootstrap honours the 238 distinct bases that underlie them, so its pooled intervals are the wider (more conservative) of the two.

**Interpretation limit.** The 300 blocks are a deterministic, provider-balanced, non-probability selection and bases are reused across providers. These intervals quantify sampling variability *within this frozen sample*. They are **not** population-representative precision statements and do not license inference to the full 12,480 provider–pair cohort.

## 6. What these results can and cannot support

**Supported by this analysis**

* A directionally consistent corruption effect and enforcement recovery on the frozen 298-block sample, in both datasets and in all six provider subgroups.
* The observation that explicitly modelling shared-base dependence does not overturn the picture given by the frozen block bootstrap.
* Descriptive, clearly bounded per-provider and provider × dataset estimates for this sample.

**Not supported by this analysis**

* A population-level six-provider semantic-effect magnitude: coverage is 2.40% of provider–pair units, OAKS Novel is absent from the semantic leg, and the selection is not a probability sample.
* Provider rankings or claims that providers differ: cells hold ~25 blocks with unstable intervals and partially shared bases.
* Validation of the frozen deterministic scorer's magnitudes: this analysis re-scores answers with the native-style evaluator and never runs that scorer.
* Hypothesis-testing significance claims: the intervals are bootstrap summaries of a deterministic non-probability sample.
* Any use of the pooled number as a single benchmark-native accuracy.
