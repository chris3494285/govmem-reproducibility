# RQ1 observability limitations

This supplementary analysis is an offline reanalysis of frozen GovMem
artifacts. No provider API was called, no model output was generated, and no
answer was re-scored.

## Verified scope

- 731 base examples.
- 2080 matched clean/corrupted/enforced fault triplets.
- 12480 provider-replicated pair observations.
- Native panel `paper_final_298`: 298 resolved Formal-300 triplets and 2 unresolved selected-panel blocks.

## State-level boundary

The frozen corruption is realised in `model_visible_memory_items` /
`model_visible_context_text`. `internal_governance_state` is unchanged and is a
copy of `oracle_state`. Consequently the injected fault label cannot serve as
an independent state detection. State anomaly prevalence and state-detector
precision, recall, false-positive rate, and false-negative rate are not
estimable. The zero internal-state-change count is retained only as a
representation diagnostic.

## Context-level interpretation

There are 2002 context changes among
2080 matched pairs. Item additions and
removals are counted relative to the evidence-supported clean projection. A
context change is not, by itself, interpreted as an absolute policy violation.

## Downstream score boundaries

Frozen proxy outcomes, LongMemEval binary native correctness, and Memora FAMA
remain separate. They are not combined into one accuracy ranking. The sanitized
provider results do not contain response text or response hashes, so actual
output identity under frozen proxy scoring is not estimable. Formal-300 does
contain answer hashes, enabling clean-versus-corrupted response-identity counts
only for resolved selected-panel blocks where both hashes exist. Response
identity is never treated as correctness identity.

The Formal-300 panel is a fixed selected validation panel. It is not a random
or representative sample of the 731-base cohort, and subgroup comparisons are
descriptive for that panel. Native panel results are not available for OAKS
Novel, so source-specific native comparisons are limited to LongMemEval and
Memora.

## Inference

Uncertainty uses a base-cluster bootstrap because multiple fault pairs share a
base. Provider-replicated observations are not pooled as independent data.
The manuscript's Formal-300 matched-block bootstrap is not replaced by this
supplementary analysis.
The matched deterministic design supports design-specific observational
contrasts only; it does not establish general causal effects or performance
rankings across different detection targets.
