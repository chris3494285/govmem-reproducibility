# B4 Paper Scorer-Impact Audit

**Verdict: `MAJOR_RERUN_REQUIRED`.** The current paper synopsis makes downstream answer quality and correctness recovery central, and those results materially depend on a scorer that has weak LongMemEval agreement and no valid Memora same-unit estimate.

## Scope limitation

No current manuscript/Prism/LaTeX source is present in this repository. `FINALIZATION_REPORT.md` independently records that omission. This audit therefore covers every substantive claim mapped by `ARTIFACTS.md`, the v3 paper-update checklist, README synopsis/headlines, and generating analyses. Exact external-paper sentence coverage remains an explicit blocker.

## What survives B4

- Cohort, source, condition, and fault-support counts.
- Governance evidence: 731/731 supported.
- Admission and model-visible context change: 12,012/12,480 (96.25%).
- Admission and context restoration: 12,480/12,480 (100%).
- Separate RQ1–RQ4 pilot artifacts, which do not consume the B4-target final-main scorer (their own evaluator validity is a separate question).

## What does not survive as semantic correctness evidence

- Event-macro corruption (−27.55 pp), recovery (+27.51 pp), condition accuracy, provider comparisons, grouped correctness effects, and correctness-based matched-fault severity.
- The Y/behavioral leg of A→C→Y propagation and claims that enforcement restores correct answers. State restoration alone does not establish correctness restoration.

## RQ-level impact

| RQ | Main metric | Uses source-aware scorer? | B4 impact | Action |
|---|---|---:|---|---|
| RQ-FM | final-main corruption/recovery and condition accuracy | yes | Core downstream-quality answer is unsupported across LongMemEval/Memora; control-plane A/C findings remain valid. | Native reevaluation or remove downstream correctness claims. |
| RQ1 | paired governance-violation delta | no | No direct B4 impact; separate deterministic evaluator. | Keep provenance separation explicit. |
| RQ2 | matched GVR/task-utility fault severity | no | No direct B4 impact; separate RQ1 scored artifacts. | Do not claim B4 validates this evaluator. |
| RQ3 | admission/context/downstream propagation in RQ1 pilot | no | No direct B4 impact; downstream metric is from a different scorer. | Keep separate from final-main A→C→Y claims. |
| RQ4 | control-plane policy tradeoffs | no | No direct B4 impact. | Retain. |
| RQ-B4 | agreement with native/native-style semantic reference | yes | LongMemEval is weak and Memora is not unit-aligned; a global scorer-validity claim fails. | Report source-specific evidence only. |

## Classification totals

- scorer_independent: 11
- scorer_dependent_primary: 5
- scorer_dependent_secondary: 2
- mixed: 3
- unclear: 1

## Remediation

Option A is the smallest scientifically defensible operation only if the paper is narrowed to control-plane/state-transition evidence. It removes central downstream-correctness conclusions. To preserve the existing downstream-quality thesis, Option D is required: reevaluate the 22,626 existing LongMemEval and Memora answer rows with frozen native/native-style protocols; no answer-generation rerun is needed. OAKS's 6,720 rows retain native option parity.

## Safety

This audit is read-only with respect to experimental inputs and made zero API, network, model, judge, scoring, or answer-generation calls.
