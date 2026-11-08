# CLEAN-A / CLEAN-B repeat-generation stochasticity control — final results

Protocol `clean-repeat-v1` · evaluator `final-native-evaluation-v2` · unit =
paired CLEAN-A / CLEAN-B block · n = **60** blocks (30 LongMemEval, 30 Memora; 10
per generating provider). No provider calls were made for this analysis.

All numbers below are produced by
`scripts/analyze_clean_repeat_stochasticity.py`, which reuses the frozen
formal-300 evaluator resolution (`resolve`) and the frozen paired percentile
bootstrap (10000 iterations, seed 20261010). CLEAN-A correctness comes from the
frozen manifest; CLEAN-B correctness is resolved from durable judge results.

## 1. Paired correctness (primary)

| Subset | n | Clean-A | Clean-B | Δ (pp) | 95% CI (pp) |
| --- | --- | --- | --- | --- | --- |
| All blocks | 60 | 0.6906 | 0.6927 | −0.21 | [−11.16, +10.75] |
| LongMemEval (binary) | 30 | 0.7333 (22/30) | 0.7000 (21/30) | +3.33 | [−16.67, +23.33] |
| Memora (native FAMA) | 30 | 0.6479 | 0.6853 | −3.75 | [−12.71, +6.25] |

Absolute gap in overall mean correctness: **0.21 pp**.

## 2. Outcome flips (binary, LongMemEval, n = 30)

Flips are only exactly defined for binary outcomes, so this is reported on the
LongMemEval subset. Transitions: correct→correct 17, correct→incorrect 5,
incorrect→correct 4, incorrect→incorrect 4. **Flips = 9/30 (30.0%)**, agreement
= 0.700.

## 3. Memora (criterion-level native FAMA, n = 30)

Correctness is continuous in [0, 1] (FAMA), so exact-equality is the meaningful
agreement notion: **24/30 blocks reproduce the exact FAMA**, mean |Δ| = 0.1041,
max |Δ| = 1.0. Binarising FAMA at 1.0 (the frozen `exploratory_binary`
projection) gives **1/30 flips**. Mean FAMA moves −3.75 pp (95% CI [−12.71,
+6.25]).

## 4. Breakdown by generating provider (10 blocks each)

| Generating provider | Clean-A | Clean-B | Δ (pp) | LongMemEval correct A→B |
| --- | --- | --- | --- | --- |
| anthropic | 0.950 | 0.850 | +10.00 | 5/5 → 5/5 |
| openai | 0.900 | 0.850 | +5.00 | 5/5 → 4/5 |
| deepseek | 0.655 | 0.660 | −0.50 | 4/5 → 4/5 |
| kimi | 0.622 | 0.779 | −15.74 | 3/5 → 4/5 |
| gemini | 0.617 | 0.650 | −3.33 | 2/5 → 2/5 |
| xai | 0.400 | 0.367 | +3.33 | 3/5 → 2/5 |

These are descriptive only (n = 10 per provider per stratum).

## 5. Comparison with the frozen formal-300 corruption effect

The frozen formal-300 downstream validation reports a corruption effect
(clean − corrupted) of **+35.54 pp, 95% CI [+29.78, +41.12]** on 298 matched
triplets. The clean-repeat paired difference of **−0.21 pp** (CI [−11.16,
+10.75]) is an order of magnitude smaller and its CI excludes the corruption
point estimate, i.e. ordinary repeat-generation stochasticity does not plausibly
account for the corruption effect at this sample size.

## 6. Scientific conclusion

**Supported.** Under the identical frozen clean input and provider
configuration, a fresh generation changes the frozen-evaluator correctness of
the paired block by a small average amount (−0.21 pp overall; +3.33 pp
LongMemEval; −3.75 pp Memora). LongMemEval outcome flips occur in 9/30 blocks,
consistent with non-trivial per-block sampling variability, but the aggregate
shift is small and its confidence interval is far below the +35.54 pp
corruption effect.

**Not supported.** This experiment does **not** establish that CLEAN-B is
equivalent to CLEAN-A, nor a bound on worst-case per-block variation (n = 30 per
source gives wide intervals, e.g. ±20 pp on LongMemEval). It measures a single
extra generation per block, not a repeat distribution, and provides no evidence
about the proposed governance/enforcement mechanism itself — only about the
stability of the clean baseline under one repeated generation. Judge completion
alone is not evidence of robustness.

## 7. Caveats

- Memora correctness is continuous native FAMA, so "flip"/"disagreement" counts
  over the 60 blocks are dominated by small FAMA differences (15/60 by exact
  float inequality); the binary flip analysis is LongMemEval-only.
- One CLEAN-B generation per block: these are one-step repeat estimates, not a
  variance decomposition, and CIs are correspondingly wide at n = 30.
- The Gemini Memora output-token override is a transport-only change
  (prompts, criteria, schema, model, and request identities unchanged); the
  frozen plan budgets are preserved in metadata.

## 8. Artifacts

- `results/stochasticity_control/clean_repeat_v1/analysis/clean_repeat_metrics.json`
- `results/stochasticity_control/clean_repeat_v1/analysis/clean_repeat_block_level.csv`
- `CLEAN_REPEAT_PAPER_TABLE.tex`
- `CLEAN_REPEAT_FINAL_AUDIT.md`
