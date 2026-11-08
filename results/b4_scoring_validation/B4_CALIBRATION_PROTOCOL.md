# B4 Semantic-Judge Calibration Protocol

## Purpose

Phase 1 is a protocol calibration and smoke test using two independent,
blinded judge families: DeepSeek and Kimi. It uses the already frozen 20-case
LongMemEval and 40-criterion Memora calibration cohort. It does not alter the
paper's final judge methodology, deterministic scores, model answers, or the
separate human-adjudication cohort.

The same frozen cases and semantic prompts are retained for a possible later
frontier-model confirmation phase.

## Precommitted interpretation rules

Agreement is calculated only over cases with valid decisions from both judges.
These thresholds were recorded before semantic execution:

- At least 90% valid A/B exact decision agreement: the protocol appears stable
  enough to proceed to frontier-model confirmation.
- At least 80% but below 90%: inspect disagreements and rubric ambiguity before
  proceeding.
- Below 80%: do not proceed to full validation; review the judge protocol first.

Regardless of agreement rate, progression is blocked by substantial parse
failures, provider failures, systematic low-confidence outputs, or obvious
rubric misunderstanding. Any such determination and its quantitative basis
must be recorded before moving to the confirmation phase.

These are calibration heuristics, not paper accuracy thresholds and not an
agreement claim about GovMem scoring.

## Execution and escalation

Each judge evaluates the same blinded semantic payload independently and never
receives the other judge's output. A case enters escalation when the judges
disagree, either returns `adjudication_required`, either has low confidence, a
response fails strict parsing, or a provider call fails.

LongMemEval retains the five-way decision contract: `correct`, `incorrect`,
`abstention_correct`, `abstention_incorrect`, and `adjudication_required`.
Memora retains the criterion-level decisions `pass`, `fail`, and
`adjudication_required`.

Raw prompts, responses, rationales, errors, and ledgers remain private. Public
reporting may include only aggregate counts, rates, safe identifiers/hashes,
and protocol metadata.
