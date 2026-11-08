# CLEAN-A / CLEAN-B Gemini Memora output-budget audit

Scope: `results/stochasticity_control/clean_repeat_v1/judging/` and
`scripts/run_clean_repeat_stochasticity.py`. No paid API calls were made. Frozen
prompts, criteria, prior results, ledgers, request identities, and the frozen
sample are unchanged.

## 1. Root cause

The clean-repeat judge plan sizes the Gemini Memora request with the frozen
formal-300 **plan** budget `40 + 36 * criterion_n`
(`scripts/prepare_formal_300_validation.py`). That budget is too small for
Gemini `gemini-3-flash-preview` on high-criterion Memora batches, which truncate
at `finish_reason = MAX_TOKENS` and emit incomplete JSON.

Observed failure (request `clean-repeat-judge::05dd5c49…78c8`, 56 criteria):

| attempt | effective budget | finish reason | raw chars | parse |
| --- | --- | --- | --- | --- |
| 1 | 1024 | (transport error) | — | malformed JSON |
| 2 | 2056 (`40+36·56`) | `FinishReason.MAX_TOKENS` | 6918 | incomplete JSON (39 `{` vs 37 `}`) |

The frozen formal-300 experiment already encountered this and established a
**transport-only recovery budget**,
`scripts/run_formal_300_validation.py::attempt3_output_budget`:

```
attempt3_output_budget(n) = min(6144, max(1024, 256 + 80 * n))
```

Its docstring records the empirical basis: the 80-token allowance exceeds the
observed successful maximum of ~69.67 output tokens per criterion, with a
256-token framing allowance, a 1024 floor, and the repository-established 6144
Gemini Flash ceiling.

## 2. Calibration against real Gemini Memora responses

Frozen formal-300 attempt-3 recovery responses
(`results/b4_scoring_validation/private/formal_300_validation_v1/gemini/attempt3_raw_responses.jsonl`,
50 rows, all `FinishReason.STOP`, budget `attempt3_output_budget(n)`):

| criteria | output tokens | budget | tokens/criterion |
| --- | --- | --- | --- |
| 6 | 414 | 1024 | 69.0 |
| 12 | 640-714 | 1216 | 53.3-59.5 |
| 33 | 1593-1872 | 2896 | 48.3-56.7 |
| 51 | 2643-2972 | 4336 | 51.8-58.3 |
| 56 | 2973-2995 | 4736 | 53.1-53.5 |
| 59 | 3189-3291 | 4976 | 54.1-55.8 |
| 86 | 4174-4798 | 6144 | 48.5-55.8 |

Clean-repeat successes at the old flat 1024 budget give the same picture
(n=3→176, n=6→342/427, n=8→534, n=13→660, n=14→845 output tokens; 50.8-71.2
tokens/criterion). The failed 56-criterion batch produced 6918 characters
(~3.4 chars/token) before hitting the 2056-token ceiling, i.e. it needed
roughly 3000+ output tokens — consistent with the frozen 4736 budget.

**Every** clean-repeat Gemini Memora request plans below the frozen transport
budget; the plan formula is under-sized for all 30 turns.

## 3. Proposed configuration (transport-only)

Add an explicit, opt-in override that raises the **Gemini Memora** effective
output budget to the frozen formal-300 transport budget:

```
--output-budget-override gemini_memora_transport
```

For a Gemini `memora_primary_gemini` request with `n` criteria the effective
budget becomes `min(6144, max(1024, 256 + 80*n))`. Nothing else changes:

- prompts, criterion lists, judgment schema, model, temperature, and request
  identities are untouched (request IDs exclude the output budget);
- the frozen formal-300 plan budget `40 + 36*n` is still stored as
  `planned_output_tokens` for audit;
- each attempt records `effective_output_tokens`, `output_budget_policy`
  (`formal300_recovery_256_80n`) and `criterion_count`;
- the override is opt-in (`--output-budget-override none` is the default), so no
  budget changes silently.

Budgets for the failed/at-risk batches under the override:

| criteria | plan (40+36n) | effective (256+80n) |
| --- | --- | --- |
| 56 (failed) | 2056 | 4736 |
| 59 (failed) | 2164 | 4976 |
| 61 | 2236 | 5136 |
| 33 | 1228 | 2896 |
| 12 | 472 | 1216 |
| ≤ 9 | ≤ 364 | 1024 |

Recommended: apply the override uniformly to every remaining Gemini Memora
request (2 failed + 22 unattempted), not just the two failures, so the whole
source shares one transport policy. The 6 already-completed Gemini Memora
judgments are left unchanged.

## 4. At-risk batches

Under the default plan budget every Gemini Memora turn is under the observed
per-criterion cost, and the highest-risk unattempted batches are `n = 61`
(plan 2236 vs. observed need ≈3.3k-4.4k tokens) and `n = 33` (plan 1228 vs.
observed ≈1.6k-1.9k). The override covers all of them.

## 5. Scientific comparability

This is a decode-budget (transport) change, not a scoring-protocol change: the
prompt, criteria, output schema, model, temperature, and the resulting judgment
format are identical, and the frozen plan budget is preserved in metadata.
Because the budget only bounds generation, a larger budget cannot change an
otherwise-complete judgment. The 6 pre-existing Gemini Memora judgments stay as
produced; recovered and newly judged batches use the larger budget, which should be noted in
the write-up as an execution-configuration detail (mirroring formal-300, which
applied the same recovery budget).

Splitting criterion batches or shortening `short_reason` is **not** done here:
either would change the scoring protocol and requires separate approval.

## 6. Remaining risks

- Gemini did not return usage metadata for the failed call, so `output_tokens`
  is unknown; the budget is sized from the frozen per-criterion calibration.
- A batch with unusually verbose `short_reason` values could still reach the
  6144 ceiling; the cap is the repository-established Gemini Flash limit.
- The override is a manual step; if it is omitted, the plan budget is used and
  truncation can recur.

## 7. Tests

Added mocked regression tests (no provider calls): explicit override metadata,
override scoping (Gemini Memora only), the frozen `attempt3_output_budget`
values, Gemini `MAX_TOKENS` truncation diagnostics, default-vs-override budget
selection, effective-budget recording per attempt, and failure-ledger
preservation across a failed recovery.

## 8. Commands

Inspect (no paid calls):

```bash
python scripts/run_clean_repeat_stochasticity.py --phase judge --dry-run
python scripts/run_clean_repeat_stochasticity.py --phase judge --recover-failed \
  --output-budget-override gemini_memora_transport --dry-run
```

Recover the two failed Gemini Memora requests with the transport budget:

```bash
python scripts/run_clean_repeat_stochasticity.py --phase judge --recover-failed \
  --judge-provider gemini --output-budget-override gemini_memora_transport \
  --max-requests 2 --execute --confirm-paid-judge-calls
```

Continue the remaining Gemini Memora primaries with the same transport budget:

```bash
python scripts/run_clean_repeat_stochasticity.py --phase judge \
  --judge-provider gemini --output-budget-override gemini_memora_transport \
  --max-requests 22 --execute --confirm-paid-judge-calls
```

Expected additional paid requests: 2 (failed recoveries) + 22 (remaining Gemini
Memora primaries) = 24, plus any result-dependent dynamic escalations the frozen
protocol later triggers.
