# Pre-registered analysis plan — CLEAN-A / CLEAN-B stochasticity control

Frozen before execution. CLEAN-B is a fresh generation under the identical frozen clean input and
provider configuration; the only difference is the generation request itself.

## Primary metrics over the 60 paired blocks

A. Clean repeat correctness disagreement rate = mean(correctness_A != correctness_B).
B. Clean repeat paired accuracy difference = mean(correctness_A - correctness_B), in percentage points.
C. Absolute clean repeat accuracy gap = |mean(correctness_A) - mean(correctness_B)|, in percentage points.
D. Transition counts: correct-correct, correct-incorrect, incorrect-correct, incorrect-incorrect.
E. Agreement rate = mean(correctness_A == correctness_B).
F. 95% paired bootstrap CI for correctness_A - correctness_B (paired over the 60 blocks, deterministic seed).

## Secondary (descriptive only)

By source and by generating provider; n is small (30 and 10), so these are descriptive.

## Comparison rule (pre-registered, descriptive)

Compare the repeat-generation variation with the frozen formal-300 corruption effect
(+35.54 pp; 95% CI 29.78-41.12). No significance threshold is invented after seeing results.
If repeat variation is much smaller than 35.54 pp, this strengthens the argument that the corruption
effect is not plausibly explained by ordinary repeat-generation stochasticity alone; if it is large,
that is reported honestly.

## Scoring protocol (unchanged from formal-300)

- LongMemEval: OpenAI gpt-5.6-sol + Anthropic claude-sonnet-5 primaries; agreement resolves;
  disagreement/non-binary routes to Gemini escalation.
- Memora: Gemini criterion-level primary; OpenAI secondary where the frozen protocol requires;
  Anthropic escalation if required; human adjudication only if the protocol reaches that state.
- The obsolete unified deterministic scorer is not used.
