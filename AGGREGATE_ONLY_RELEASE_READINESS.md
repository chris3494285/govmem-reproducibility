# GovMem v3 aggregate-only release readiness

Scope: a conservative, **aggregate-only**, clean-history-ready snapshot of the
GovMem final v3 experiment, the B4 scoring validation, the formal-300
validation, and the clean-repeat stochasticity control.

Preparation date: 2026-10-14. No model/API call, no experiment rerun, no change
to any frozen manifest, ledger, result, or experiment conclusion. No manuscript
file was modified. No Git commit or push was performed.

This supersedes the earlier Tier-B readiness note (`PUBLIC_RELEASE_READINESS.md`
in the developer tree): Tier-B removed per-example identifiers and per-row
fingerprints but still shipped row-level score files carrying a row/condition
structure derived from the licensed benchmarks. Those files are **not** shipped
here.

## Verdict (read first)

- **Aggregate-only.** The release ships no row-level result file: no
  condition-level score file, no per-instance/per-base table, no
  cohort/judge-request/validation dump, no persistence ledger, and no
  generation/judging row. Only aggregate tables/summaries, protocol and
  readiness documentation, code, offline tests, and non-secret configs remain.
- **Published metrics retained.** The final-v3, B4, formal-300, and
  clean-repeat aggregates are all present (see the retention table below).
- **No private mapping.** No original→pseudonym mapping is required or shipped;
  no retained file contains a source-derived identifier, so the Tier-B mapping
  is not used.
- **Content audit clean.** Build audit (`build_public_release_candidate.py`)
  and independent audit (`audit_public_release_candidate.py` + an ad-hoc scan):
  0 data-artifact violations, 0 credentials, 0 email addresses, 0 machine
  paths, no `.git`. After the pre-publication cleanup, only **three** production
  script files carry benchmark-derived tokens (documented, fail-closed, see §5);
  no public test embeds one any more.
- **Public offline suite green.** 568 passed / 253 skipped / 0 failed on the
  candidate.
- **Still not "safe to publish" without one decision.** As in the Tier-B note,
  redistribution rights for the three licensed benchmarks are not established
  by this repository. Aggregate-only shrinks the footprint to group-level
  statistics, but the authors must still confirm that publishing those derived
  aggregate facts is permitted (see §9).

## 1. Candidate location and shape

```
/tmp/govmem-aggregate-only-release-v3
```

(On this macOS host `/tmp` resolves to `/private/tmp`.) The directory contains
the release files plus three manifest files: `MANIFEST.sha256`,
`FILE_MANIFEST.json`, `FILE_MANIFEST.csv`.

| Property | Value |
|---|---|
| Release files (including this readiness file, excluding the 3 manifest files) | **406** |
| Total bytes (release files) | **4,360,581** |
| Machine-path-sanitized files | 16 |
| Content violations | 0 |
| `.git` directory | absent |
| Private id mapping | not present |

Files by category:

| Category | Files |
|---|---:|
| `scripts` | 119 |
| `tests` | 104 |
| `v3_analysis` | 46 |
| `package/eval` | 36 |
| `package/govmem` | 20 |
| `b4_scoring` | 18 |
| `configs` | 15 |
| `root` | 15 |
| `formal_300` | 11 |
| `clean_repeat` | 7 |
| `docs` | 4 |
| `package/baselines` | 4 |
| `third_party` | 4 |
| `frozen_metadata` | 3 |

(The 3 manifest files are additional and are not counted above.)

## 2. How the candidate is produced

The candidate is produced from a **deny-by-default allowlist**, never by
copying the repository:

```bash
python scripts/build_public_release_candidate.py \
  --repo-root . \
  --allowlist release/aggregate_only_release_allowlist.json \
  --output /tmp/govmem-aggregate-only-release-v3 --force
```

`release/aggregate_only_release_allowlist.json` is the machine-readable policy:
an explicit `include` set, an explicit `exclude` set, path-sanitization rules,
the content `forbidden_patterns`, reviewed `audit_exemptions`, and the rationale
for every row-level exclusion. The builder fails closed: if any selected file
still tripped a forbidden pattern, nothing would be copied.

No `--id-mapping` is passed: this release does not pseudonymize, because no
retained file contains a source-derived identifier.

## 3. What is retained (published metrics)

Every headline number is preserved as an aggregate:

| Experiment | Retained aggregate artifacts | Headline values |
|---|---|---|
| Final v3 (six providers) | `results/final_main_cross_provider_summary_v3/*`, `results/answer_leakage_repair/final_v3_analysis/` (provider summaries, per-event/fault/provider/source tables, control-plane diagnostic, matched-fault support, governance-evidence audit, old-vs-new statistics) | corruption −27.55 pp, recovery +27.51 pp, context-changed −26.04 pp, context-unchanged −1.28 pp, control-plane change 96.25%, restoration 100% |
| B4 scoring validation | `results/b4_scoring_validation/` summaries: `b4_validation_summary.md`, judge-request summaries, `oaks_input_summary.json`, scorer-validation summary+manifest, unit-alignment summaries, paper-impact audit | cohort/condition counts, dedup judge-case counts, scorer agreement distributions, paper-impact counts |
| Formal-300 validation | `results/formal_300_validation_v1/`: protocol, remaining-failure audit, `analysis*/formal_validation_summary.json`, `dry_run_summary.json`, `sampling_lock.json`, recovery summaries | resolved matched triplets 297/300; `strong_positive` signal; by-fault/by-source/by-provider effects |
| Clean-repeat control | `results/stochasticity_control/clean_repeat_v1/`: `analysis/clean_repeat_metrics.json`, `clean_repeat_block_level.csv`, `clean_repeat_paper_table.tex`, plan/preflight/summary | n_blocks 60, block agreement, by-source/by-provider metrics |

The clean-repeat block table is keyed by **release-synthetic** labels of the
form `clean-repeat-v1::<provider>::<source>::NN`. Those labels are generated by
the release protocol and identify no benchmark example; the block→example
mapping is not shipped.

## 4. What is excluded

| Excluded | Reason |
|---|---|
| `results/final_main_tier_b/**` (6 row files + manifest) | Row-level score files (requirement 2). |
| `results/final_main_sanitized/**` | Superseded row-level score files, retained identifiers/fingerprints. |
| `results/answer_leakage_repair/changed_conditions.jsonl`, `changed_condition_ids.txt`, `**/mechanism_instance_level.csv`, `**/governance_evidence_base_level.csv`, `**/matched_fault_instance_level.csv`, `governance_evidence_spotcheck.jsonl`, `source_evidence_provenance.json` | Per-example/per-instance/per-base rows; source-derived ids and per-row fingerprints. |
| `results/stochasticity_control/clean_repeat_v1/manifest.jsonl`, `generation_*.jsonl`, `judging/**`, `manifest_replacement_audit.json` | Per-block/per-request rows and content-hash judge ids. |
| `results/formal_300_validation_v1/**/*.jsonl` and `recovery_attempt*/attempt*_authorization*.json` | Per-observation/per-slot rows and per-slot fingerprints. |
| `results/b4_scoring_validation/*_cohort.jsonl`, `*_cohort_manifest.json`, `*_judge_requests.jsonl`, `*_validation.json`, `b4_validation_summary.json`, `memora_*` row files, `oaks_native_validation_input.jsonl`, scorer/unit-alignment per-case JSONL, `private/**` | Per-case rows, native judge payloads, and per-case ids. |
| `data/*.jsonl` benchmark manifests, oracle dumps, provider rerun manifests, `m3_intervention_example`, the manuscript (`paper_aiopen_final/`, the paper zip), historical/exploratory result directories, `.git`, caches, logs, `.env` | Deny-by-default; not matched by any `include` pattern. |

Also excluded after review (each contained at least one concrete source-derived
identifier): `CLEAN_REPEAT_FINAL_AUDIT.md` (one real LongMemEval base id) and
`data/system_validation_frozen_v1_metadata.json` (concrete GateMem
episode/checkpoint ids). Both remain in the developer tree only.

## 5. Content audit

Two independent checks were run over the candidate:

1. `scripts/audit_public_release_candidate.py` — verifies no `.git`, that every
   file matches its recorded SHA-256, that no file is unlisted, and that every
   file passes the allowlist `forbidden_patterns` (including the added
   `source_derived_identifier`, `block_or_slot_fingerprint`, `row_identity_key`,
   and expanded `content_fingerprint_field` rules). Result: **PASS, 406/406
   files, 0 violations.**
2. An independent ad-hoc scan for benchmark identifiers, block/slot
   fingerprints, fingerprint fields, row-identity keys, credentials, email
   addresses, and machine-specific paths.

| Check | Data artifacts (`json/jsonl/csv/md/txt/tex`) | Code (`.py`) |
|---|---|---|
| Source-derived benchmark ids | 0 | 2 files, reviewed (below) |
| Block/slot content fingerprints | 0 | 1 file, reviewed (below) |
| Per-row content-fingerprint fields | 0 | n/a (code) |
| Row-identity keys | 0 | n/a (code) |
| Benchmark question/answer/context text keys | 1 reviewed synthetic fixture | n/a (code) |
| Credential shapes / emails / user paths | 0 / 0 / 0 | 0 / 0 / 0 |
| `.git` | absent | — |

**Reviewed synthetic fixture.** `tests/fixtures/memora-mini/**` is an
author-created three-session fixture that mimics the Memora question/answer
schema with invented content. It contains no benchmark-derived content and no
real identifiers; the benchmark-text key rule is exempted for it
(`audit_exemptions`, reviewed 2026-10-14).

**Public tests no longer embed identifiers.** A pre-publication pass replaced
every real identifier in the public test suite: the Kimi and Grok targeted
selection tests now derive the condition id from the local frozen manifest, the
restrictive-scoring RQ1b test selects its MemConflict pair structurally, the B4
recovery test imports the production `TARGET_STARTED_ONLY` constant, and the
formal-300 tests import the frozen slot constants and derive the
2048-ceiling slot ids from the live ledger. See `FINAL_GITHUB_RELEASE_AUDIT.md`
§1 for the file-by-file record.

**Reviewed production constants (the only remaining residual).** Three
production scripts keep a benchmark-derived value because it identifies a
specific frozen artifact and cannot be replaced without changing frozen
experiment semantics. Each is documented in place and fails closed (raises on
drift/missing input). They are exempted by rule with a per-file reason in the
allowlist:

| File | Residual | Fail-closed guard |
|---|---|---|
| `scripts/audit_paper_evidence_counts.py` | 1 Memora base id (`M3_BASE_ID`, the paper's M3 example) | `build_m3()` raises if the base or any condition is absent |
| `scripts/authorize_and_resume_final_b4_holdout.py` | 1 B4 holdout judge-request id (`TARGET_STARTED_ONLY`) | `verify_recovery_targets()` raises on any other started-only target |
| `scripts/run_formal_300_validation.py` | 3 formal-300 slot fingerprints (recovery-protocol targets) | authorization validators raise on manifest/ledger drift |

No literal identifier is repeated in this document by design (that would itself
place an identifier in a shipped file).

## 6. Reproducibility tiers

Full mapping in `docs/REPRODUCTION_TIERS.md`:

- **Tier 1 — verifiable from this release.** All aggregate tables and summaries
  (final v3, B4, formal-300, clean-repeat) and the code/tests that read them.
- **Tier 2 — reproducible only after independently obtaining the licensed
  datasets and the original local experiment outputs.** Rebuilding the frozen
  manifests and recomputing the aggregates end to end; B4 native/semantic
  counts; formal-300 resolution (also needs `results/cheap_judge_pilot_v1/`);
  clean-repeat generation/judging ledgers.
- **Tier 3 — not fully reproducible from this release alone.** Any per-example
  analysis, per-row fingerprint verification, the formal-300 test module
  (imports the excluded execution plan), and manuscript-dependent tests.

This release does **not** claim complete reproducibility: the row-level outputs
and the licensed inputs they were derived from are required for Tiers 2–3 and
are not shipped.

## 7. Tests

| Run | Result |
|---|---|
| Developer tree — public offline suite | 787 passed, 293 skipped, 0 failed (re-measured) |
| Candidate — public offline suite (`PYTHONPATH=. python -m pytest`) | **568 passed, 253 skipped, 0 failed** |

The candidate collects 821 tests. The decrease versus the Tier-B candidate
(which collected ~1,080) is `tests/test_formal_300_validation.py` (~259 tests):
its module reads the excluded `results/formal_300_validation_v1/execution_plan.jsonl`
at import time, so `tests/conftest.py` now ignores that one module when its
import-time input is absent. On the developer tree the input is present and the
module is collected and run exactly as before (hence the developer-tree total
of 1,080 collected tests).

Four tests that previously passed only because the Tier-B candidate shipped
row-level files (`test_b4_validation.py` scorer-join and unit-alignment checks;
`test_clean_repeat_stochasticity.py` manifest-shape and replacement-audit
checks) are now scoped `requires_private_artifacts` in
`tests/public_release_test_scope.json`; they were observed failing on a fresh
candidate run and now skip by default. Consequently the developer-tree default
run reports four more skips than the previous baseline (787/293 versus
791/289); `pytest --include-private` still runs all four.

## 8. Exact verification commands

```bash
# 1. Build the aggregate-only candidate (deny-by-default allowlist; no mapping)
python scripts/build_public_release_candidate.py \
  --repo-root . --allowlist release/aggregate_only_release_allowlist.json \
  --output /tmp/govmem-aggregate-only-release-v3 --force

# 2. Independent structure + content audit
python scripts/audit_public_release_candidate.py /tmp/govmem-aggregate-only-release-v3 \
  --allowlist release/aggregate_only_release_allowlist.json

# 3. Checksum verification
( cd /tmp/govmem-aggregate-only-release-v3 && shasum -a 256 -c MANIFEST.sha256 )

# 4. Public offline test suite from inside the snapshot
( cd /tmp/govmem-aggregate-only-release-v3 && PYTHONPATH=. python -m pytest )
```

Observed: build `406 files / 0 violations`; audit `PASS`; `shasum -c`
`406/406 OK, 0 non-OK`; candidate suite `568 passed, 253 skipped, 0 failed`.

## 9. Remaining blockers and decisions

1. **Licensing (blocking).** Confirm redistribution rights for the licensed
   benchmarks, here limited to *derived aggregate facts* (dataset-name labels,
   GovMem taxonomy labels, and group-level statistics). Aggregate-only is the
   smallest reasonable footprint, but it is not a licence. If aggregates are
   not permitted, withhold the whole `results/` tree.
2. **Clean history (required).** Publish this `.git`-free snapshot into a **new
   repository with a single fresh commit**. No history was rewritten here.
3. **Code-embedded identifiers (decision).** Three production scripts keep one
   frozen-artifact identifier each (§5); they are documented and fail closed.
   Either accept them as reviewed frozen-protocol constants, or move them behind
   a private configuration before an even stricter release. The public tests no
   longer embed any identifier.
4. **Reduced formal-300 test coverage.** ~259 formal-300 tests cannot be
   collected inside the snapshot because their module needs the excluded
   execution plan. They are Tier 3 in `docs/REPRODUCTION_TIERS.md`.
5. **Manuscript- and legacy-dependent tests.** `test_paper_revision_source`
   (external manuscript) and `test_main_dataset_audit` (legacy MemConflict
   manifest) remain out of scope for the public suite.
6. **Upstream pilot artifacts.** `results/cheap_judge_pilot_v1/**` stays
   excluded; formal-300 inputs are reproducible only with those local files.

## 10. Safe-to-publish summary

| Question | Answer |
|---|---|
| Are row-level score files shipped? | **No** — aggregate-only. |
| Are source-derived per-example identifiers in shipped *data artifacts*? | **No**. |
| Are per-row content fingerprints in shipped *data artifacts*? | **No**. |
| Are the published aggregates for v3 / B4 / formal-300 / clean-repeat retained? | **Yes** (group-level only). |
| Is a private id mapping required or shipped? | **No / No**. |
| Every file has a SHA-256? | **Yes** (406/406, `shasum -c`). |
| Public offline suite green? | **Yes** (568 / 253 / 0). |
| No `.git`, credentials, PII, or machine paths? | **Yes**. |
| Benchmark-derived tokens in shipped *code*? | **3 production constants only** (§5) — documented, fail-closed; open decision #3. |
| Safe to publish **today**? | **No — one licensing decision open.** Aggregate-only limits redistribution risk to derived aggregate facts but does not establish the right to publish them. |
| Clean-history-ready? | **Yes** — no `.git`; publish as a fresh repository. |
