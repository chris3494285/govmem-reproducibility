# B4 Semantic Judge Protocol Audit

Status: offline protocol audit only. No API, network, model-generation, or
judge calls were made. No final judge decisions were generated.

## 1. Executive findings

OAKS is ready for native option-parity reporting: 6,720 rows were supplied,
6,714 resolved, and all 6,714 resolved rows agreed. Six rows remain
ambiguous/unscored.

The LongMemEval plan has 96 deduplicated cases and an explicit decision
category list, but the request payload has two protocol weaknesses:

1. provider and clean/corrupted/enforced condition metadata are present in the
   judge payload and should be removed from the judge-facing prompt for blind
   evaluation;
2. the request has the question, answer, and reference answer, but no supplied
   memory/context or explicit per-item answerability metadata. The reference
   answer alone is not sufficient evidence for a judge to distinguish a
   justified information-insufficient response from an unjustified refusal in
   every case.

The Memora paid-judge request set has 508 records, but the payload is not yet
judge-ready. It contains answer hashes and criterion hashes, not the model
answer or criterion wording/expected fact. A judge cannot independently
evaluate the atomic criterion from the current request file. This is a clear
structural preparation bug, not a scoring-semantic change.

The 419 preference-recommendation cases remain a separate adjudication queue.
They must not be silently added to the paid judge pool.

## 2. LongMemEval input audit

### Current request fields

`longmemeval_judge_requests.jsonl` contains:

- `request_id`
- `question_id`
- `question`
- `reference_answer`
- `answer`
- `answer_hash`
- `native_abstention_answers`
- `decision: null`
- `decision_schema`
- `condition`
- `provider`
- provenance containing provider, condition, and B4 validation ID
- `execution: dry_run_only`

### Field disposition

| Field | Judge-facing? | Finding |
|---|---:|---|
| Benchmark question | Yes | Required. |
| Native/reference answer | Yes | Required for native-style comparison, but should be identified as reference material rather than a target label. |
| Existing model answer | Yes | Required. |
| Native abstention conventions | Yes, carefully | Useful, but should be accompanied by per-item answerability/context evidence. |
| Question/context answerability evidence | Yes | Missing from the current request. Add relevant local/private memory/context or a benchmark-native answerability field. |
| Provider | No | Provenance only; hide from the judge prompt. |
| Clean/corrupted/enforced condition | No | Hide from the judge prompt; retain only outside the prompt for stratified analysis. |
| GovMem deterministic score | No | Not present, which is correct. |
| Previous scorer decision | No | Not present, which is correct. |
| Desired agreement/statistics | No | Not present, which is correct. |
| Other judge decisions | No | Not present, which is correct. |

The current file is therefore not leaking GovMem correctness or desired
agreement, but it is not fully blind because provider and condition metadata are
present. More importantly, it does not contain enough answerability evidence
for robust abstention judgments. The private judge payload should include the
minimum relevant benchmark context or native answerability annotation while
keeping the public manifest ID/hash-only.

### Proposed LongMemEval rubric

The judge should evaluate semantic content, not wording, style, or verbosity.

- `correct`: the answer gives the native answer, or a semantically equivalent
  answer, for an answerable instance; it has no material contradiction or
  unsupported extra claim.
- `incorrect`: the answer gives a materially wrong answer, contradicts the
  reference/context, answers a different question, or refuses an answerable
  instance.
- `abstention_correct`: the answer explicitly indicates insufficient
  information, and the supplied benchmark context shows that the instance is
  not answerable at that point.
- `abstention_incorrect`: the answer abstains/refuses even though the supplied
  benchmark context supports a determinate answer.
- `adjudication_required`: the context, reference, or answer is ambiguous;
  multiple interpretations remain defensible; or the answer contains a
  borderline semantic contradiction that cannot be resolved reliably.

The judge must not treat lexical mismatch as failure when the meaning is
equivalent. It must not accept an answer that adds a material unsupported claim
or contradiction merely because it contains the reference phrase.

### Proposed LongMemEval judge schema

```json
{
  "request_id": "...",
  "decision": "correct | incorrect | abstention_correct | abstention_incorrect | adjudication_required",
  "confidence": "high | medium | low",
  "short_reason": "one or two sentences",
  "answerability_basis": "answerable | insufficient_information | ambiguous"
}
```

The judge prompt should receive only:

- the benchmark question;
- relevant benchmark memory/context or native answerability evidence;
- the native/reference answer;
- the existing model answer;
- the abstention convention, if applicable.

It should not receive provider, model, condition, GovMem score, or aggregate
B4 information.

Current plan counts are 100 cohort rows and 96 deduplicated question/answer
cases. The request summary correctly records 35 clean, 35 corrupted, 30
enforced, and 20 rows per provider. Those labels should remain analysis-only.

## 3. Memora input audit

### Current paid-judge request fields

Each of the 508 records contains:

- `request_id`
- provider-aware `criterion_id` and `sample_id`
- original condition ID
- provider and condition
- subtype
- `answer_sha256`
- `criterion_text_sha256`
- a short generic rubric
- deduplicated criterion IDs/provenance
- `judge_count: 3`
- `execution: dry_run_only`

The corresponding `memora_atomic_criteria.jsonl` contains criterion type,
expected behavior, native question ID, subtype, and criterion text hash. It
does not contain the criterion wording or expected fact. The request file also
does not contain the model answer, only its hash. Consequently, a judge cannot
evaluate one criterion independently from the current paid request artifact.

This is the principal structural bug found in this audit.

### Required private judge payload

The smallest repair should construct a separate private execution payload by
joining each deduplicated request to local cohort/criterion data. It should
contain:

- relevant user/context information needed for the criterion;
- the exact atomic criterion wording or native field/fact;
- expected behavior (`yes`/`no` or the native field constraint);
- the existing model answer;
- a provider-blind request ID.

The public request/manifest may retain hashes and IDs only. The execution
payload must not contain GovMem deterministic scores, previous pass/fail
results, B4 aggregates, other judge outputs, provider identity, or condition
labels unless a documented semantic necessity is established.

### Leakage result

The current Memora request does not leak a deterministic score or desired
agreement outcome. It does expose provider and condition metadata, which should
be removed from the judge-facing prompt. Its larger problem is omission rather
than leakage: hashes are not sufficient judging inputs.

## 4. Proposed Memora rubric

Each judge evaluates exactly one atomic criterion and returns:

```json
{
  "criterion_id": "...",
  "decision": "pass | fail | adjudication_required",
  "confidence": "high | medium | low",
  "short_reason": "one or two sentences"
}
```

For `content_project_proposal` and `content_email_writeup`:

- `pass` only when the answer semantically satisfies the specified native
  fact/constraint;
- `fail` when the required information is absent, materially contradicted,
  replaced with a different fact, or only vaguely gestured at;
- `adjudication_required` when the criterion/evidence is ambiguous or the
  interpretation is genuinely subjective.

Harmless paraphrase passes. Verbosity and general writing quality are ignored.
The judge does not assign a holistic quality score.

## 5. Three-judge aggregation

Three calls to the same model are not three genuinely independent judgments.
If three judges are retained, they should be separately instantiated and,
preferably, heterogeneous.

Recommended decision handling:

- `PASS/PASS/PASS`, `FAIL/FAIL/FAIL`: consensus; accept.
- `PASS/PASS/FAIL`: accept provisional pass and record disagreement; no final
  paper label until the disagreement policy is applied.
- `FAIL/FAIL/PASS`: accept provisional fail and record disagreement; same final
  review requirement.
- `PASS/FAIL/ADJUDICATION_REQUIRED`: human adjudication required.
- `PASS/PASS/ADJUDICATION_REQUIRED`: provisional pass, but retain in human
  adjudication because one judge could not resolve it.
- `FAIL/FAIL/ADJUDICATION_REQUIRED`: provisional fail, but retain in human
  adjudication.
- Any multiple `ADJUDICATION_REQUIRED` outputs: human adjudication required.

No `adjudication_required` output should be silently converted into a final
pass/fail by majority vote.

For LongMemEval, the preferred economical design is one blind judge followed
by escalation of low-confidence, abstention-borderline, and disagreement cases.
Three same-model calls are not methodologically attractive for this small
cohort.

## 6. Judge-model independence

Using an answer-generation provider as judge risks self-preference, stylistic
preference, and correlated errors. Provider/model identity should be hidden from
the prompt. It should remain in external provenance only.

One judge model is acceptable for an initial protocol if a blinded human
calibration subset is reported. Three repeated calls to the same model should
not be described as three independent judges.

Recommended architecture:

1. two heterogeneous, blinded judge families for Memora semantic cases;
2. automatic agreement when both return the same pass/fail decision;
3. a third heterogeneous tie-breaker only on disagreement;
4. human review for any adjudication-required output or unresolved tie.

This is stronger than three calls to the same model while avoiding a third call
for every case. Exact call count depends on the observed disagreement rate and
must not be fabricated before execution.

## 7. Treatment of the 419 adjudication cases

The 419 cases are preference-recommendation `memory_presence` criteria. They
are borderline because satisfying a preference criterion requires assessing a
concrete recommendation, compatibility with likes/dislikes, and constraints
such as already-visited or forbidden items. They should remain a separate
human/native adjudication queue for the primary paper result.

Recommended human protocol:

- two blinded annotators per case;
- hide provider, model, condition, GovMem score, and other annotator outputs;
- show the native criterion/context and model answer only;
- use `pass`, `fail`, and `adjudication_required`;
- send disagreements and low-confidence cases to a third expert adjudicator;
- report raw agreement and Cohen's kappa for the two primary annotators;
- if the design uses more than two raters on all cases, report nominal
  Krippendorff's alpha or Fleiss' kappa instead.

An LLM may assist triage after human calibration, but it should not replace
this human protocol without demonstrating agreement on a blinded calibration
set. These 419 cases must not enter the 508 paid judge pool automatically.

## 8. Recommended validation statistics

### OAKS

Report total, resolved, ambiguous/unscored, disagreements, and raw agreement
among resolved cases. OAKS has native option labels, so kappa is unnecessary.
A Wilson binomial confidence interval for resolved agreement is reasonable.

### LongMemEval

Report cohort rows, deduplicated judgment cases, native-style agreement,
disagreements, correct/incorrect counts, abstention-correct and
abstention-incorrect counts, adjudication count, and unscored count. Report
raw agreement after adjudication. Cohen's kappa is appropriate only if GovMem
and the adjudicated native-style result are represented as the same nominal
categories. Otherwise report the confusion table and raw agreement rather than
forcing kappa.

### Memora

Report total atomic criteria, native-exact criteria, judge-required criteria
before and after deduplication, adjudication criteria, judged pass/fail counts,
disagreements with GovMem, and unresolved cases. For the final binary
pass/fail comparison after adjudication, raw agreement plus Cohen's kappa is
appropriate. For multiple human/judge raters, report nominal Krippendorff's
alpha or Fleiss' kappa, alongside raw agreement and the adjudication rate.

Confidence intervals should be reported for primary agreement proportions;
bootstrap intervals are suitable for clustered criteria, provided the sampling
unit is stated. Criteria from the same sample should not be treated as fully
independent observations without qualification.

## 9. Cost-efficient protocol comparison

The following counts exclude any unobserved disagreement escalation.

| Design | Memora calls | LongMemEval calls | Comment |
|---|---:|---:|---|
| A: 3 same-model judges | 1,524 | 96 | Current three-call plan, but repeated same-model calls are correlated. |
| B: 2 heterogeneous + tie-breaker | 1,016 + tie-breakers | 96 + escalation | Tie-breakers are charged only for disagreements; range is 1,016–1,524 for Memora before human review. |
| C: one strong judge | 508 | 96 | Lowest cost, but requires a blinded human calibration/audit subset and has weaker independent validation. |

Recommended: Design B for the 508 Memora semantic cases, with human review of
all adjudication-required outputs and a blinded calibration subset. Use one
blind LongMemEval judge with escalation rather than three repeated same-model
calls. No API prices are assumed.

## 10. Exact remaining steps before B4 is complete

1. Repair the private LongMemEval execution payload to include answerability
   context/metadata and remove provider/condition from the judge-facing prompt.
2. Repair the private Memora execution payload to include criterion wording,
   expected behavior, relevant context, and model answer; retain hashes in
   public-safe artifacts.
3. Pre-register the rubrics, blind fields, aggregation rules, and escalation
   policy.
4. Run blinded human calibration for the 419 preference-recommendation
   adjudication cases and report inter-annotator agreement.
5. Execute the approved native/judge validation only with explicit authority;
   no execution occurred in this audit.
6. Compare GovMem deterministic outcomes with adjudicated native-style labels
   and publish source-stratified agreement statistics with unresolved cases
   shown separately.

B4 is not complete until those semantic/native labels and adjudications exist.
