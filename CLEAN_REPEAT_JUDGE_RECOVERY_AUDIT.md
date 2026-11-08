# CLEAN-A / CLEAN-B judge execution & recovery audit

Scope: `results/stochasticity_control/clean_repeat_v1/judging/` and
`scripts/run_clean_repeat_stochasticity.py`. No paid API calls were made during
this audit. No CLEAN-A/CLEAN-B generation, frozen sample, formal-300 artifact,
or paper file was modified. Existing ledger, result, and diagnostic history was
read only (append-only artifacts were not rewritten or truncated).

## 1. Root causes

The clean-repeat judge executor was written with a **flat** output-token budget
(`JUDGE_OUTPUT_TOKENS = {"longmemeval": 160, "memora": 1024}`) and did **not**
apply the frozen formal-300 output-budget rules. The frozen formal-300 protocol
(`scripts/prepare_formal_300_validation.py`) plans LongMemEval at 160 tokens but
plans Memora at `40 + 36 * criterion_n`, and
`scripts/run_formal_300_validation.py::formal_output_budget` applies transport
reliability floors (Anthropic always >= 2048; OpenAI LongMemEval primary >= 512).

Every one of the five failures is explained by this omission:

| provider | slot | block | planned (old) | frozen budget | failure |
| --- | --- | --- | --- | --- | --- |
| gemini | memora_primary_gemini | clean-repeat-v1::deepseek::memora::56 | 1024 | 2056 (criterion_n=56) | malformed JSON (unterminated string @ char 3425) |
| gemini | memora_primary_gemini | clean-repeat-v1::kimi::memora::37 | 1024 | 2164 (criterion_n=59) | malformed JSON (unterminated string @ char 3722) |
| openai | lme_primary_openai | clean-repeat-v1::gemini::longmemeval::23 | 160 | 512 (floor) | response contained no text |
| anthropic | lme_primary_anthropic | clean-repeat-v1::gemini::longmemeval::23 | 160 | 2048 (floor) | response contained no text |
| anthropic | lme_primary_anthropic | clean-repeat-v1::openai::longmemeval::04 | 160 | 2048 (floor) | malformed JSON (unterminated string @ char 58) |

The two Gemini Memora failures truncated at ~3.4k-3.7k characters with an
unterminated JSON string, exactly what a ~1024-token budget produces for a
56-59-criterion batch. The two Anthropic and one OpenAI failures are consistent
with the frozen reliability floors (thinking/latency consuming a 160-token
budget before text). This is a transport/configuration omission, **not** a
scoring-semantics change.

Secondary cause: the failure-recording path discarded the safe transport
diagnostics captured by the adapters (`exc.diagnostics` / `adapter.last_diagnostics`
`/ adapter.last_metadata`), so the five failures recorded only `error_class` and
`error_message`. There was no finish reason, token usage, response length, or
truncation signal to confirm the diagnosis from the artifacts alone.

## 2. Files modified

- `scripts/run_clean_repeat_stochasticity.py`
  - explicit ledger/result **state machine** (`reconcile_judge_state`) keyed by
    request ID and attempt number, with fail-closed handling for ambiguous or
    interrupted attempts and explicit invalid/conflicting categories;
  - **safe explicit recovery** mode (`--recover-failed`) with `--max-requests`,
    `--execute`, `--confirm-paid-judge-calls`, plan-membership validation, and
    incremented attempt numbers;
  - frozen formal-300 **output-budget** alignment (`planned_output_tokens`,
    `judge_output_budget`, `--judge-output-budget {formal300,planned}`);
  - richer, sanitized **failure diagnostics** (`judge_failure_diagnostics`);
  - tolerant JSONL reading (`read_jsonl_tolerant`) that skips and reports blank
    or corrupt lines without dropping history;
  - a `judge_reconciliation.json` audit artifact and an optional offline
    recovery path (`judge_recovery_candidates.jsonl`).
- `tests/test_clean_repeat_stochasticity.py` — regression tests for the state
  machine, recovery gating/bounding/idempotency, diagnostics, and budgets.
- `results/stochasticity_control/clean_repeat_v1/judging/judge_reconciliation.json`
  (new, derived audit artifact).
- This report: `CLEAN_REPEAT_JUDGE_RECOVERY_AUDIT.md`.

## 3. Existing ledger/result reconciliation

Artifacts (unchanged, read only):

- `judge_execution_ledger.jsonl`: 36 events (18 started, 13 completed, 5 failed),
  18 unique request IDs, all at attempt 1. The file begins with one pre-existing
  blank line (tolerated and reported, not deleted).
- `judge_results.jsonl`: 13 durable results (also preceded by one blank line).
- `judge_diagnostics.jsonl`: 18 records (13 completed, 5 failed).

State machine result (`judge_reconciliation.json`):

```
completed_with_result : 13
failed                :  5
unattempted           : 70
fail_closed           :  0
recoverable           :  5
```

Categories produced by `reconcile_judge_state`:

- `completed_with_result` — terminal `completed` event **and** a durable result.
- `failed` — terminal `failed` event, no result. A recorded failure is a
  **terminal** state, never an interrupted attempt.
- `started_without_terminal` — a `started` attempt with no later terminal event
  (interrupted/ambiguous) -> **fail-closed**.
- `completed_without_result` — terminal `completed` with no durable result ->
  **fail-closed**.
- `result_without_terminal` / `failed_with_result` — orphan or inconsistent
  result records -> **fail-closed**.
- `duplicate_completed_events`, `terminal_without_started`, `conflicting_identity`
  — invalid/conflicting -> **fail-closed**.

Completion is never inferred from API usage or a successful transport alone: a
request is `completed_with_result` only when the frozen parser produced a durable
`parsed_result`.

## 4. Detailed status of the five failed requests

All five are terminal `failed` at attempt 1 and **recoverable** (never treated as
interrupted). None has a durable result.

| # | request id | provider / model | slot | block |
| --- | --- | --- | --- | --- |
| 1 | `clean-repeat-judge::05dd5c49...78c8` | gemini / gemini-3-flash-preview | memora_primary_gemini | clean-repeat-v1::deepseek::memora::56 |
| 2 | `clean-repeat-judge::098d21bd...7030` | openai / gpt-5.6-sol | lme_primary_openai | clean-repeat-v1::gemini::longmemeval::23 |
| 3 | `clean-repeat-judge::0a4cadeb...ab79` | gemini / gemini-3-flash-preview | memora_primary_gemini | clean-repeat-v1::kimi::memora::37 |
| 4 | `clean-repeat-judge::0accce48...0baa` | anthropic / claude-sonnet-5 | lme_primary_anthropic | clean-repeat-v1::gemini::longmemeval::23 |
| 5 | `clean-repeat-judge::0e07af61...71b0` | anthropic / claude-sonnet-5 | lme_primary_anthropic | clean-repeat-v1::openai::longmemeval::04 |

Request identity (provider, model, source, prompt hash and all frozen inputs) is
unchanged; recovery preserves it and only increments the attempt number.

## 5. Offline recoverability of historical responses

**None of the five historical responses can be recovered offline.** The failure
records contain no response text (the malformed-JSON and empty-text payloads were
never persisted; only `response_hash` is stored for successes). Verified: the
five `judge_diagnostics.jsonl` failure rows carry no `raw_text_preview`,
`raw_text_tail`, or `json_parse_ok` fields. A dedicated offline recovery hook
(`judge_recovery_candidates.jsonl` -> `_offline_recover`) now exists and is
preferred whenever a raw artifact is present, but it is a no-op for the current
five failures, which require paid re-execution.

## 6. Proposed token-budget adjustments

Minimal, evidence-backed, and aligned with the frozen formal-300 protocol (no
prompt, schema, model, or decoding-temperature change):

1. **Memora (Gemini)**: plan `40 + 36 * criterion_n` instead of a flat 1024
   (-> 2056 and 2164 for the two failed observations). Source: frozen
   `scripts/prepare_formal_300_validation.py`.
2. **LongMemEval OpenAI primary**: apply the formal floor 512
   (`FORMAL_300_OPENAI_LME_PRIMARY_MIN_OUTPUT_TOKENS`).
3. **LongMemEval Anthropic primary**: apply the formal floor 2048
   (`FORMAL_300_ANTHROPIC_MIN_OUTPUT_TOKENS`).

These are applied by default through `--judge-output-budget formal300`; the stored
planned budget is not mutated by the floors, and `--judge-output-budget planned`
reproduces the previous behaviour for comparison. The request IDs, prompts, and
model are unchanged, so this is a transport-reliability configuration change
only.

## 7. Tests performed

- `python -m pytest -q tests/test_clean_repeat_stochasticity.py` — 46 passed
  (including: started->completed, started->failed, started-without-terminal,
  duplicate completed, completed-without-result, failed->recovered attempt 2,
  repeated recovery, completed never re-selected, recovery confirmation/bound
  gating, malformed JSON, empty-text, missing/inconsistent results, interrupted
  recovery fail-closed, budget floors). All provider calls are mocked.
- `python -m pytest -q` — see section 9 (57 pre-existing, unrelated,
  environment failures; the clean-repeat suite passes).
- `git diff --check` — clean.
- Dry-run / audit commands (section 8) exercise reconciliation with zero paid
  calls.

## 8. Safe commands

Inspect (no paid calls):

```bash
python scripts/run_clean_repeat_stochasticity.py --phase judge --dry-run
python -m json.tool results/stochasticity_control/clean_repeat_v1/judging/judge_reconciliation.json
python scripts/run_clean_repeat_stochasticity.py --phase judge --recover-failed --dry-run
```

Recover the five failed requests (paid; exactly five calls, then result-dependent
escalations only if the frozen protocol triggers them):

```bash
python scripts/run_clean_repeat_stochasticity.py --phase judge --recover-failed \
  --max-requests 5 --execute --confirm-paid-judge-calls
```

Two-request canary before the full recovery:

```bash
python scripts/run_clean_repeat_stochasticity.py --phase judge --recover-failed \
  --max-requests 2 --execute --confirm-paid-judge-calls
```

## 9. Expected additional paid request counts

- Recovery of the five failed requests: **5 paid requests** (one per failed
  request, at attempt 2).
- No dynamic escalations are currently required; additional result-dependent
  escalations materialize only after primaries complete, per the frozen
  protocol.
- `PAID_CALLS_PERFORMED=0` for every dry-run/audit command.

Repository test status: `python -m pytest -q` reports **57 failed, 1000 passed,
10 skipped**. All 57 failures are pre-existing and environmental — they live in
`test_final_main_*`, `test_answer_leakage`, `test_b4_validation`,
`test_control_plane_propagation`, `test_matched_fault_support`,
`test_sanitize_final_results`, `test_system_validation_paper_artifacts`, and
`test_formal_300_validation`, and each raises `FileNotFoundError` for local
artifacts absent from this workspace (e.g.
`data/final_main_governance_condition_manifest_v1.jsonl`,
`configs/final_main_token_budget_kimi_v1.json`,
`results/final_main_deepseek_v4_pro/scored_results.jsonl`,
`results/system_validation_statistics/system_validation_statistics_by_system.csv`).
None of the failing test modules imports
`scripts/run_clean_repeat_stochasticity.py`, and no clean-repeat test failed.

## 10. Remaining scientific limitations

- The 13 already-completed judgments were produced at the old flat budgets and
  are intentionally left unchanged; the 5 recovered judgments will use the
  frozen formal-300 budgets. Prompts, schema, model, and decoding temperature are
  identical, so correctness semantics are unchanged, but the execution-budget
  difference across the two batches should be disclosed in any write-up.
- Recovery re-runs the same model/prompt per request; it does not combine or
  average repeated judgments, so the inferential unit (one judgment per
  request) is unchanged.
- If a recovered attempt also fails, the request remains `failed` (recoverable)
  and requires a human decision; it is never silently retried.
- No offline recovery was possible for the current failures because the raw
  response text was not persisted; only future failures can benefit from that
  path.
