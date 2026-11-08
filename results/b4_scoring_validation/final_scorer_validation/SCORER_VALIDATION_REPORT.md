# B4 Deterministic Scorer Validation

The frozen GovMem source-aware deterministic scorer is compared with the fully adjudicated independent semantic reference. No scoring or judging is executed by this analysis.

## Mapping and orientation

- Independent LongMemEval `correct`/`incorrect` and Memora `pass`/`fail` map to binary correct/incorrect.
- Frozen deterministic `correctness=true/false` maps to correct/incorrect only when `scoring_status=scored`.
- Positive means correct. FP means scorer-correct/reference-incorrect; FN means scorer-incorrect/reference-correct.
- LongMemEval is answer-level versus answer-level. Memora compares the frozen answer-level deterministic score with each atomic criterion; the answer-level score is repeated for criteria sharing that answer.

## Overall

- N: 604
- agreement: 274
- disagreement: 330
- agreement rate: 45.36%
- Cohen's kappa: 0.114259
- false positives: 19
- false negatives: 311
- Wilson 95% CI: [0.414357, 0.493514]

## By task

| Task | N | Agreement | Disagreement | Rate | Kappa | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| longmemeval | 96 | 60 | 36 | 62.50% | 0.258687 | 12 | 24 |
| memora | 508 | 214 | 294 | 42.13% | 0.111962 | 7 | 287 |

## By memora subtype

| Group | N | Agreement | Disagreement | Rate |
|---|---:|---:|---:|---:|
| content_email_writeup | 79 | 36 | 43 | 45.57% |
| content_project_proposal | 429 | 178 | 251 | 41.49% |

## By generating provider

| Group | N | Agreement | Disagreement | Rate |
|---|---:|---:|---:|---:|
| openai | 132 | 32 | 100 | 24.24% |
| anthropic | 153 | 47 | 106 | 30.72% |
| gemini | 66 | 35 | 31 | 53.03% |
| deepseek | 0 | 0 | 0 | not available |
| kimi | 164 | 87 | 77 | 53.05% |
| xai | 89 | 73 | 16 | 82.02% |

## By experimental condition

| Group | N | Agreement | Disagreement | Rate |
|---|---:|---:|---:|---:|
| clean | 166 | 83 | 83 | 50.00% |
| corrupted | 247 | 108 | 139 | 43.72% |
| enforced | 191 | 83 | 108 | 43.46% |

## By adjudication resolution method

| Group | N | Agreement | Disagreement | Rate |
|---|---:|---:|---:|---:|
| human_adjudication | 1 | 1 | 0 | 100.00% |
| primary_binary_agreement_raw_subtype_disagreement | 6 | 4 | 2 | 66.67% |
| primary_exact_agreement | 580 | 264 | 316 | 45.52% |
| third_judge_majority | 17 | 5 | 12 | 29.41% |

DeepSeek has N=0 because it was not part of the frozen 604-case semantic cohort; no provider identity is inferred.

Disagreement records contain IDs, labels, scorer metadata, and immutable provenance only—no benchmark question, reference, or model-answer text.
