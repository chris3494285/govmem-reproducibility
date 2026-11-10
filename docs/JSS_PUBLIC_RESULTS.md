# Public results and reproduction scope

This document records the aggregate evidence included in the public GovMem
reproducibility package. It contains no benchmark text, per-example
identifier, answer hash, prompt, model response, judge output, or human
adjudication record.

## Frozen cohort

| Quantity | Value |
|---|---:|
| Unique base examples | 731 |
| LongMemEval bases | 191 |
| Memora bases | 380 |
| OAKS Novel bases | 160 |
| Supported base-fault pairs | 2,080 |
| Clean conditions | 731 |
| Corrupted conditions | 2,080 |
| Enforced conditions | 2,080 |
| Unique condition IDs | 4,891 |
| Evaluated generating providers | 6 |
| Provider-replicated pair evaluations | 12,480 |

Fault-pair support reconciles as:

- missing: 731;
- stale: 351;
- flipped: 380;
- conflicting: 618.

## Research-question mapping

The labels below follow the final Applied Intelligence manuscript, not the
historical `rq1`–`rq4` prefixes used by earlier pilot artifacts.

| Applied Intelligence research question | Evidence in the manuscript | Public implementation | Public aggregate evidence |
|---|---|---|---|
| **RQ1: Agent sensitivity and observability** | Paired projection changes, selected native-style outcomes, and full-panel deterministic-proxy screening | `govmem/governance/`, `scripts/run_final_main_*`, `scripts/analyze_formal_300_subgroups.py`, `scripts/finalize_formal_300_validation.py`, `scripts/analyze_jss_rq1_observability.py`, `scripts/analyze_control_plane_propagation.py`, `scripts/finalize_v3_offline.py` | `results/formal_300_subgroup_v1/`, `results/jss_public_aggregate_v1/rq1_observability_v2.json`, `results/answer_leakage_repair/final_v3_analysis/control_plane_propagation/`, `results/final_main_cross_provider_summary_v3/` |
| **RQ1 supplementary: Context–output observability** | Separation of operative-context change from selected downstream failure events and answer-identity changes | `scripts/analyze_jss_rq1_observability.py`, `scripts/audit_jss_rq1_observability.py`, `scripts/build_jss_public_aggregate.py` | `results/jss_rq1_observability_v2/`, `results/jss_public_aggregate_v1/rq1_observability_v2.json` |
| **RQ2: Fault coverage and severity** | Supported fault eligibility, per-fault proxy contrasts, and same-base stale-versus-missing/stale-versus-conflicting comparisons | `scripts/analyze_source_event_fault_distribution.py`, `scripts/analyze_matched_fault_support.py`, `scripts/analyze_final_main_results.py` | `results/source_event_fault_distribution_v1/`, `results/answer_leakage_repair/final_v3_analysis/matched_fault_support/`, `results/final_main_cross_provider_summary_v3/` |
| **RQ3: Cross-model and cohort dependence** | Benchmark-source, event-category, and provider strata, including source–event confounding and dependence-aware native-panel summaries | `scripts/analyze_formal_300_subgroups.py`, `scripts/analyze_final_main_results.py`, `scripts/analyze_source_event_fault_distribution.py` | `results/formal_300_subgroup_v1/`, `results/source_event_fault_distribution_v1/`, `results/final_main_cross_provider_summary_v3/` |
| **RQ4: Evaluation and reference-restoration boundaries** | Oracle-restored sibling behavior, native-style versus proxy scorer checks, identical-answer inconsistency, and the clean-repeat control | `scripts/analyze_control_plane_propagation.py`, `scripts/analyze_formal_300_subgroups.py`, `scripts/analyze_jss_scoring_consistency.py`, `scripts/analyze_clean_repeat_stochasticity.py`, `scripts/analyze_b4_scorer_validation.py`, `scripts/validate_oaks_parity.py` | `results/answer_leakage_repair/final_v3_analysis/control_plane_propagation/`, `results/formal_300_subgroup_v1/`, `results/jss_public_aggregate_v1/scoring_consistency_sensitivity.json`, `results/stochasticity_control/clean_repeat_v1/analysis/`, `results/b4_scoring_validation/final_scorer_validation/`, `results/b4_scoring_validation/unit_alignment_audit/`, `results/b4_scoring_validation/oaks_input_summary.json` |

The selected native-style outcome estimates use the manuscript's final
298-resolved-triplet panel: 148 LongMemEval binary-correctness triplets and
150 Memora FAMA triplets. The public aggregate files preserve the reported
counts and summary statistics, but end-to-end recomputation of that selected
panel requires the separately retained private records.

The older `scripts/build_rq1_paper_table.py`,
`scripts/analyze_rq2_fault_severity.py`,
`scripts/analyze_rq3_error_propagation.py`, and
`scripts/evaluate_rq4_enforcement_policies.py` concern separate historical
pilot, propagation, or enforcement-policy analyses. Their filenames and output
paths are preserved as provenance, but their historical RQ numbering does not
define the Applied Intelligence RQ mapping above.

## Full-cohort control plane

Across 12,480 provider-replicated pair evaluations:

- operative admission/model-visible projection changed under corruption in
  12,012 cases: **96.25%**;
- oracle-backed enforcement restored the clean projection in 12,480 cases:
  **100%**.

This is an intervention-fidelity result. It is not downstream accuracy and
does not imply autonomous fault detection or repair.

## Formal-300 behavioral validation

The formal panel contains 300 selected matched provider-by-base blocks and 900
observations. The final human-adjudicated analysis resolves 298 complete
triplets over 238 distinct bases; two LongMemEval observations remain
unresolved and are not imputed.

| Quantity | Final value |
|---|---:|
| Resolved triplets | 298 |
| Resolved distinct bases | 238 |
| Clean mean | 0.6903 |
| Corrupted mean | 0.3349 |
| Enforced mean | 0.6617 |
| Corruption effect | +35.54 pp |
| Corruption 95% CI | [29.78, 41.12] |
| Recovery effect | +32.67 pp |
| Recovery 95% CI | [26.94, 38.36] |
| Restoration gap | -2.86 pp |
| Restoration-gap 95% CI | [-5.94, 0.12] |

The native pooled intervals use a paired matched-block bootstrap with 10,000
resamples and seed `20261010`. The source-specific and dependence-aware
intervals use a paired base-cluster bootstrap with 10,000 resamples, the same
seed, and 238 resolved base clusters. The two procedures answer different
questions and are not interchangeable.

The shipped non-adjudicated public analysis resolves 297 triplets. That value
must not be substituted for the final 298-triplet human-adjudicated result.

## RQ1 supplementary: Context–output observability (Observability V2)

The Applied Intelligence manuscript's RQ1 supplementary analysis separates
state, model-visible context, and downstream outcomes.

### Context projection

| Quantity | Result |
|---|---:|
| Internal governance state changed versus clean | 0 / 2,080 |
| Independent structural state anomalies | 0 / 2,080 |
| Operative context changed | 2,002 / 2,080 |
| Corrupted context added an item relative to clean | 1,080 / 2,080 |
| Corrupted context removed an item relative to clean | 1,462 / 2,080 |
| Enforcement restored the clean context | 2,080 / 2,080 |

State-detector precision, recall, false-positive rate, and false-negative rate
are not estimable. The frozen corruption is projection-level, internal
governance state remains unchanged, and using the injected fault label as a
detector result would be circular.

### Native Formal-300 output observability

| Surface | Result |
|---|---:|
| LongMemEval clean-to-corrupted response identity changed | 102 / 148 |
| LongMemEval response identity unchanged | 46 / 148 |
| LongMemEval response identity unchanged but score changed | 1 / 148 |
| LongMemEval observed induced-failure event | 64 / 148 |
| Memora response identity changed | 132 / 150 |
| Memora response identity unchanged | 18 / 150 |
| Memora response identity unchanged but score changed | 0 / 150 |
| Memora observed FAMA-decrease event | 96 / 150 |

RQ1 V2 uses a base-cluster bootstrap with 5,000 resamples and seed `20261101`.
Its intervals are separate from the Formal-300 10,000-resample analyses.

## Isolated scoring inconsistency

One of the 148 resolved LongMemEval triplets has identical clean, corrupted,
and enforced answer identities and identical evaluator inputs, but the clean
condition received score 1 while corrupted and enforced received score 0.
The frozen results are retained.

A common-score sensitivity that assigns all three identical conditions score 0
changes:

- LongMemEval corruption from +40.54 to +39.86 pp, a **0.68 pp** change;
- LongMemEval recovery from +39.19 to +39.19 pp, unchanged;
- pooled corruption from +35.54 to +35.20 pp, a **0.34 pp** change;
- pooled recovery, unchanged.

No sign, qualitative conclusion, or resolved-triplet count changes. The shipped
aggregate at
`results/jss_public_aggregate_v1/scoring_consistency_sensitivity.json` records
the complete aggregate sensitivity without source identifiers or answer hashes.

## Reproduction limits

Directly verifiable from the package:

- every reported aggregate number;
- cohort and condition counts;
- provider, source, fault, and event strata;
- bootstrap settings and aggregate intervals;
- the scoring-consistency counts and common-score sensitivity;
- the absence of benchmark text and row-level experiment records.

Requires licensed datasets:

- rebuilding the frozen manifests;
- recomputing provider-condition context hashes and source eligibility;
- rerunning source-derived per-example analyses.

Requires private experiment records:

- recomputing the final 298/898 Formal-300 analysis;
- recovering provider response, judge, and adjudication evidence;
- reproducing the row-level scoring-inconsistency trace.

The package therefore does not claim full end-to-end reproducibility.
