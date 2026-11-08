# Formal-300 Gemini Remaining-Failure Audit

Audit: `formal-300-gemini-remaining-failure-audit-v1`. Protocol: `formal-300-validation-v1`. Evaluator: `final-native-evaluation-v2`. This report is entirely offline and contains no prompt, answer, question, criterion text, or raw provider response.

## Current state

- Successful durable results: 399
- `INVALID_ARGUMENT`: 45
- Malformed JSON: 6
- Duplicate durable results: 0

## Request-shape comparison

| Group | Metric | Min | P50 | P90 | Max |
|---|---|---:|---:|---:|---:|
| Success | Criteria | 2 | 11 | 17 | 21 |
| Success | Prompt characters | 1079 | 3685 | 7607 | 12958 |
| Success | Estimated input tokens | 1079 | 3685 | 7607 | 12970 |
| Success | Schema characters | 543 | 1025 | 1519 | 1807 |
| Success | Planned output tokens | 112 | 436 | 652 | 796 |
| Success | Answer characters | 7 | 303 | 1797.4 | 6385 |
| Invalid argument | Criteria | 33 | 54 | 69 | 86 |
| Invalid argument | Prompt characters | 7647 | 13120 | 15089 | 18833 |
| Invalid argument | Estimated input tokens | 7647 | 13122 | 15089 | 18833 |
| Invalid argument | Schema characters | 1920 | 3201 | 3528 | 3978 |
| Invalid argument | Planned output tokens | 1228 | 1984 | 2524 | 3136 |
| Invalid argument | Answer characters | 16 | 195 | 580.4 | 854 |
| Malformed JSON | Criteria | 6 | 10 | 14.5 | 17 |
| Malformed JSON | Prompt characters | 2737 | 3363 | 6742 | 7883 |
| Malformed JSON | Estimated input tokens | 2737 | 3363 | 6743 | 7885 |
| Malformed JSON | Schema characters | 814 | 980.5 | 1488.5 | 1655 |
| Malformed JSON | Planned output tokens | 256 | 400 | 562 | 652 |
| Malformed JSON | Answer characters | 114 | 544.5 | 1057 | 1391 |

## Invalid-argument diagnosis

The groups are perfectly separated by criterion count in this snapshot: successes end at 21 criteria, while every rejected request has at least 33. Schema sizes are likewise separated (1807 versus 1920 characters). Prompt sizes overlap, so prompt length alone does not explain the failures.

The schema uses one array-item object, not one property per criterion. However, it embeds every criterion ID in an enum and sets `minItems=maxItems=criterion_count`. Criterion IDs are all nonempty and unique, and none use unusual characters. Gemini returned only a generic 400, so offline evidence cannot distinguish an enum-cardinality constraint from an array-bound or aggregate schema-complexity constraint.

A scientifically neutral repair is available: use a generic strict item schema with `criterion_id` as a string, then retain the existing downstream validator's exact count, exact ID-set, duplicate-ID, unknown-ID, decision-label, confidence, and reason checks. Removing transport-level ID enumeration and cardinality encoding changes no prompt, criterion, judgment semantics, or FAMA aggregation.

## Malformed-JSON diagnosis

The six cases have criterion counts [6, 8, 8, 12, 12, 17], planned output budgets [256, 328, 328, 472, 472, 652], and parser failure positions [935, 1006, 1068, 1539, 1659, 2141]. Each response ended inside a JSON string near its output envelope, which strongly supports truncation. Nevertheless, raw outputs, usage, finish reasons, and the actual batch-level adapter budget were not durably preserved, so truncation cannot be proven from provider metadata.

An output-budget change is warranted for any future explicitly authorized recovery of these six cases, but it should be frozen and tested independently rather than inferred as a blanket semantic change.

## Recovery status

All 51 latest failures consumed attempt 2. They are not automatically retryable. A future recovery is technically possible only after the strict generic schema and audited output-budget policy are frozen and a new explicit append-only authorization permits another attempt. No recovery was authorized or executed here.

## Conclusions

- Invalid-argument root cause: response_schema_complexity_tracks criterion count perfectly; criterion-ID enum cardinality and array min/max cardinality are confounded, and Gemini did not expose its exact internal constraint.
- Malformed-JSON root cause: probable output truncation: all errors end inside a JSON string near the planned output envelope, but raw output, usage, finish reason, and actual batch-level max_output_tokens were not preserved.
- Methodology change required: no.
- API calls performed: 0.
