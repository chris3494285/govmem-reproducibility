# Memora Unit-Alignment Audit

- Atomic semantic cases: 508
- Unique generated answers: 42
- Answer–criterion assignments: 511
- Current repeated answer-to-criterion comparison valid: no
- Valid alignment from frozen artifacts: none
- New judge/API calls used: no
- Scorer rerun used: no

## Frozen units

GovMem scorer: one generated answer; Boolean correct only when every extracted content_data leaf fact is deterministically supported.

Independent reference: one native evaluation sub-question/atomic criterion; native benchmark aggregates these as accuracy components and per-question FAMA.

## Path 1 — aggregate criteria to answer level

The native benchmark defines per-question FAMA from memory-presence and forgetting-absence accuracies, not a frozen Boolean all-criteria-pass label. GovMem's Boolean scorer covers extracted content_data facts and does not encode the native forgetting penalty, so an all-pass or any-pass projection would invent a new target.

## Path 2 — recover deterministic criterion predictions

Frozen scorer details expose content_data field paths and support bits, but B4 criteria expose native evaluation-question IDs/types and hashes without an immutable field-to-criterion mapping. The criteria also include forgetting_absence items absent from scorer field details.

Only 4 of 42 answers even have matching scorer-field and native memory-presence criterion counts; count equality still would not prove identity.

## Conclusion

Path 3 applies: neither proposed alignment is defensible from the frozen artifacts. No corrected Memora accuracy or revised cross-dataset overall metric is computed. LongMemEval remains a valid standalone answer-level comparison.
