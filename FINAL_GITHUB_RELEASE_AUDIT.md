# GovMem aggregate-only release — final pre-publication audit

Audit date: 2026-10-14. Auditor: automated offline audit run in the
`paper-v3-public` worktree.

Constraints honoured: **no paid calls, no experiment reruns, no changes to
frozen data, no manuscript edits, no git commit or push.** Every command in this
audit is offline and read-only with respect to frozen artifacts. The only files
written are the public test/script sources this audit was asked to clean up, the
two audit documents, and the release candidate directory under `/tmp`.

## Verdict

**NOT PUBLISHABLE AS-IS.** One blocking decision remains: redistribution rights
for the three licensed benchmarks, including the derived *labels and group-level
statistics* this release still carries, are not established by this repository.

Everything the repository can verify about itself passes: the candidate builds
from an explicit allowlist, the content audit is clean on all data artifacts,
all SHA-256s verify, and the public offline suite is green (568 passed / 253
skipped / 0 failed). The only residual benchmark-derived tokens are three
documented, fail-closed production constants. See §7 for the exact publish
conditions.

## 1. Code files carrying benchmark-derived identifiers

The previous candidate had **eleven** code files with benchmark-derived tokens
and exempted all of them. This audit inspected each one and reduced the set to
**three production scripts**. The audit rule itself was also narrowed from
coarse dataset-prefix matching (`memora::<any letter>`, `memconflict::`, …) to
the concrete real-identifier shapes, which removed six false-positive
exemptions (synthetic fixtures and f-string prefixes) and means the rule now
fires only on genuine identifiers.

### 1a. Public tests — real identifiers replaced (pass)

| File | Was | Now |
|---|---|---|
| `tests/test_final_main_kimi.py` | 1 literal Memora condition id | derived from the local frozen manifest by dataset/condition/fault/budget-class (`_frozen_condition_id`) |
| `tests/test_final_main_grok.py` | 1 literal LongMemEval base id | derived from the local frozen manifest (`_frozen_condition_id`) |
| `tests/test_governance_fault_rq1_restrictive_scoring.py` | 1 literal MemConflict example id | structural selector over the frozen file (source/dimension/attribute) with a length assertion |
| `tests/test_b4_validation.py` | 1 literal B4 holdout judge id | imports the production `TARGET_STARTED_ONLY` constant; the synthetic `memora`-prefixed placeholder stayed as-is |
| `tests/test_formal_300_validation.py` | 4 literals of the attempt-4 smoke slot + 3 hardcoded 2048-ceiling slot ids | attempt-4 slot imported as `ATTEMPT4_SMOKE_REQUEST_ID`; the 2048-ceiling ids are now derived from the live ledger (`_token_limit_2048_ids`), independent of the selector under test; the driver helper uses a synthetic id |

The two files with only false-positive matches
(`tests/test_frozen_rq1a_semantics.py`, `scripts/build_b4_validation_artifacts.py`)
needed no code change once the rule was narrowed, and their exemptions were
removed.

Verification: the five edited test files were run individually with
`--include-private` on the developer tree — **all pass** —
plus the full `tests/test_formal_300_validation.py` module (all pass) and
`tests/test_paper_evidence_audit.py` (all pass). No test semantics were weakened;
the derived ids are read from the same frozen inputs the tests already required.

### 1b. Production constants — kept, documented, fail-closed (accepted residual)

These identify a specific frozen artifact; replacing them would change which
frozen experiment is reproduced, so they were kept and hardened instead.

| File | Constant | Why it cannot be synthetic | Fail-closed behaviour |
|---|---|---|---|
| `scripts/audit_paper_evidence_counts.py` | `M3_BASE_ID` (paper M3 example base) | the audit reports that exact frozen base | `build_m3()` now raises if the base or any of the three conditions is missing |
| `scripts/authorize_and_resume_final_b4_holdout.py` | `TARGET_STARTED_ONLY` (holdout judge id) | only that exact request is the frozen started-only target | `verify_recovery_targets()` raises if any other row claims that reason |
| `scripts/run_formal_300_validation.py` | 3 formal-300 slot fingerprints | they are the frozen recovery-protocol targets | authorization validators raise on manifest/ledger drift |

Each was given an in-file comment stating the dependency, and each already (or
now) aborts rather than proceeding when its frozen input is absent or drifted.
They remain the only `source_derived_identifier` / `block_or_slot_fingerprint`
exemptions in the allowlist.

## 2. Formal-300: 297 vs 298 matched triplets — reconciled

Both numbers describe the **same frozen 300 blocks** (same
`frozen_block_fingerprint`, protocol `formal-300-validation-v1`, evaluator
`final-native-evaluation-v2`). They differ by **one block** because they come
from **two different analysis drivers / versions**, not from a statistic change.

| | Public analysis (shipped) | Private final analysis (not shipped) |
|---|---|---|
| Artifact | `results/formal_300_validation_v1/analysis/formal_validation_summary.json` | `results/b4_scoring_validation/private/formal_300_validation_v1/final_analysis/formal_300_final_metrics.json` |
| Driver | `scripts/analyze_formal_300_validation.py` | `scripts/finalize_formal_300_validation.py` |
| Human adjudication | **not applied** | **applied** (1 recorded blinded adjudication) |
| Unresolved observations | 3 | 2 |
| Resolved matched triplets | **297 / 300** | **298 / 300** |
| clean / corrupted | 0.6896 / 0.3330 | 0.6903 / 0.3349 |
| corruption effect | +35.66 pp, CI [+30.05, +41.34] | +35.54 pp, CI [+29.78, +41.12] |
| signal | `strong_positive` | `strong_positive` |

The single differing unit is one Memora / Kimi / conflicting block: its
`enforced` observation is the one resolved by explicit blinded human
adjudication. The public driver does not read the human-adjudication record, so
that block is excluded there (3 unresolved → 297); the private final driver
does, so it is included (2 unresolved → 298).

**Which number the clean-repeat work uses.** The clean-repeat comparator
("Formal-300 corruption effect", +35.54 pp on 298 triplets) in
`CLEAN_REPEAT_FINAL_RESULTS.md`, `CLEAN_REPEAT_PAPER_TABLE.tex`, and the
`formal_300_comparator` block of `clean_repeat_metrics.json` is the **private
final (298)** version — its clean/corrupted values and CI match
`formal_300_final_metrics.json` exactly.

**Release implication.** A reader with only this release can recompute the 297
value from the shipped summary (Tier 1) but cannot reproduce the 298 value,
because its source artifact is private and is not shipped. This is a
provenance/labelling gap, not a numerical error, and **no statistic was
altered**. Recommendation: when the manuscript is next revised, add a one-line
provenance note to the clean-repeat table stating that its comparator is the
human-adjudicated final analysis (298/300), distinct from the shipped
non-adjudicated public summary (297/300). This audit deliberately does not edit
any result file.

## 3. Aggregate-only policy review of the shipped files

Every shipped `results/` file was inspected for granularity, key type, and
identifier/fingerprint content.

**Determination: no shipped file is source-derived per-example data.**

| File (shipped) | Granularity | Key | Verdict |
|---|---|---|---|
| `results/stochasticity_control/clean_repeat_v1/analysis/clean_repeat_block_level.csv` | 60 rows = designed sampling blocks (10 examples each; mapping not shipped) | release-synthetic `clean-repeat-v1::<provider>::<source>::NN` | **Block-level aggregate, not per-example.** Columns are dataset-name/provider/fault/route labels plus per-block CLEAN-A/CLEAN-B correctness. No benchmark id, no fingerprint, no text. |
| `results/stochasticity_control/clean_repeat_v1/analysis/clean_repeat_metrics.json` | same 60 blocks embedded in `blocks` + overall/by-source/by-provider aggregates | synthetic block ids | same as above |
| `.../clean_repeat_paper_table.tex`, `analysis_plan.md`, `preflight_report.md`, `manifest_summary.json`, `expected_calls_and_cost.json` | aggregate/summary | – | aggregate; `manifest_summary.json` reports a *count* of unique clean condition ids, not the ids |
| `results/formal_300_validation_v1/analysis*/formal_validation_summary.json` | 300-block aggregate | by source/provider/fault | aggregate |
| `results/formal_300_validation_v1/{dry_run_summary,sampling_lock,gemini_remaining_failure_audit,recovery_attempt*/attempt*_recovery_summary}.json` | aggregate + whole-file provenance hashes | – | aggregate/provenance (placeholder or hash only) |
| `results/final_main_cross_provider_summary_v3/*` | 6-row provider table | provider/model | aggregate |
| `results/answer_leakage_repair/final_v3_analysis/**` (`mechanism_by_*`, `*_summary.json`, `provider_summaries.json`, `providers/*/analysis_summary.json`, `providers/*/analysis_table.csv`, `old_vs_new_statistics.*`) | group-level | provider/source/event/fault/statistic | aggregate; `provider_result_paths` / `result_path` are repo-relative path strings, not ids |
| `results/answer_leakage_repair/{answer_leakage_audit_v3,repair_summary,rerun_cost_estimate,rerun_identity_audit}.json` | aggregate counts + hashes | – | `affected_base_ids`, `affected_rows`, `unresolved_source_provenance_condition_ids` are all **empty lists** |
| `results/b4_scoring_validation/*` (shipped subset) | distributions, counts, provenance | – | aggregate; scorer/unit summaries are confusion-matrix and distribution tables |
| `results/answer_leakage_repair/final_v3_analysis/licensing_blocker_report.{json,md}` | licence analysis | – | documents which excluded files exposed which fields |

Two exclusions that this review re-confirmed as necessary: the row-level Tier-B
score files and the per-instance/per-base CSVs (all absent from the candidate),
and `data/system_validation_frozen_v1_metadata.json` (concrete GateMem
episode/checkpoint ids; absent).

## 4. Redistribution concerns and unresolved permission questions

No shipped file contains benchmark text, a per-example benchmark identifier, or
a per-example content fingerprint. What the release **does** still carry, and
whose redistribution rights the authors must confirm, is:

1. **Dataset-name labels** (`longmemeval`, `memora`, `oaks_novel`, `gatemem`,
   `memconflict`) as `source`/`source_dataset` values and in prose.
2. **Derived structural facts**: cohort sizes (731 bases; 191/380/160 by source;
   4,891 condition rows; 12,480 control-plane comparisons; 1111/2660/1120 rows
   per provider by source), the OAKS source-question count (435), and per-source
   row counts in `paper_impact_summary.json`.
3. **Taxonomy labels**: the GovMem fault/event/condition taxonomy
   (missing/stale/flipped/conflicting; deletion/temporal/update;
   clean/corrupted/enforced) and the **Memora native task-subtype and
   criterion labels** (`activity_todos`, `content_email_writeup`,
   `content_project_proposal`, `preference_recommendation`, `memory_presence`,
   `forgetting_absence`) in `memora_judge_request_summary.json`,
   `scorer_validation_summary.json`, `memora_unit_alignment_audit.json`, and the
   provider token-budget configs.
4. **Group-level statistics** (accuracies, effects, CIs, agreement rates)
   keyed by the labels above.
5. **Whole-file provenance hashes** of licensed-derived manifests and answer
   artifacts (`*.freeze.json`, `sampling_lock.json`, `scorer_validation_manifest.json`,
   `rerun_identity_audit.json`). These are hashes, not content, but they do
   fingerprint the licensed inputs.
6. **Schema/filename documentation** in `docs/OBTAINING_DATASETS.md` (expected
   field names and the official source filenames).
7. **The clean-repeat block table** (§3): the sampling design (6 providers × 2
   sources × 5 blocks) and per-block correctness.

Unresolved permission questions, stated plainly:

- Does each benchmark licence permit publishing **derived aggregate statistics
  and labels** (items 1–5, 7), even with all per-example content removed?
- Does any licence restrict publishing **hashes of the licensed source file or
  of manifests derived from it** (item 5)?
- Does the Memora licence permit publishing its **native subtype / criterion
  taxonomy names** (item 3)?
- Does any licence restrict **naming the source files** in acquisition docs
  (item 6)?

None of these is answerable from inside the repository. They are the single
blocking decision.

## 5. Release candidate

Built with the existing explicit allowlist, unchanged in structure:

```bash
python scripts/build_public_release_candidate.py \
  --repo-root . \
  --allowlist release/aggregate_only_release_allowlist.json \
  --output /tmp/govmem-aggregate-only-release-v3 --force
```

```
/tmp/govmem-aggregate-only-release-v3   (400 files incl. this audit; no .git; no private mapping)
```

## 6. Verification results

| Check | Command | Result |
|---|---|---|
| Build | `build_public_release_candidate.py … --force` | **PASS** — 400 files, 0 violations |
| Structure + content audit | `audit_public_release_candidate.py <cand> --allowlist …` | **PASS** — 400 files on disk, 400 manifest entries, **0 content violations** |
| SHA-256 | `shasum -a 256 -c MANIFEST.sha256` | **PASS** — 400/400 OK, 0 non-OK |
| Public offline suite (candidate) | `PYTHONPATH=. python -m pytest -p no:cacheprovider` | **PASS** — 568 passed, 253 skipped, **0 failed** (821 collected) |
| Public offline suite (developer tree) | same | **PASS** — 787 passed, 293 skipped, 0 failed |
| Independent content scan (all files) | regex scan for ids, fingerprints, credentials, emails, machine paths | **PASS** — 0 data-artifact hits; 3 production-constant code hits (§1b) |
| `.git` / private mapping in candidate | filesystem check | **absent / absent** |

Note on the independent scan: it also matches `.py` files that contain the
literal *names* of fingerprint/row-identity keys (code that reads or writes
fields such as a context hash key or a condition-id key). Those are field-name
references in source code, not per-example data; the allowlist applies the
content-fingerprint and row-identity rules to data files only, and no data
artifact in the candidate contains such a field.

Fail/fail-closed coverage: the only "fail" outcomes in this audit are the
documented fail-closed guards, which abort when a frozen artifact is missing or
drifted. No verification step was skipped.

## 7. Publish recommendation

| Question | Answer |
|---|---|
| Is the candidate internally consistent and free of forbidden content? | **Yes** — audit PASS, 0 violations. |
| Are all SHA-256 checksums correct? | **Yes** — 400/400. |
| Does the public offline suite pass? | **Yes** — 568/253/0. |
| Do any shipped files contain benchmark text? | **No.** |
| Do any shipped files contain per-example identifiers or fingerprints? | **No** (data artifacts), and no public test does either. |
| Do shipped files contain derived aggregate facts / labels? | **Yes** — see §4. |
| Any code-embedded identifiers? | **3 production constants**, documented and fail-closed (§1b). |
| **Publish now?** | **NO — one blocking decision.** |

**Recommendation: do not publish until the §4 permission questions are
answered.** If the authors confirm that derived aggregate statistics, dataset
labels, taxonomies, and provenance hashes may be published, then publish this
candidate as a fresh single-commit repository (it has no `.git` history).

If that confirmation is not obtained, the safe fallback is to publish only the
code, tests, docs, and configs, and withhold or further coarsen the `results/`
aggregates.

Two non-blocking follow-ups for the authors:

1. Add the §2 provenance note distinguishing the 297 (public) and 298
   (human-adjudicated, private) Formal-300 denominators.
2. Decide whether to keep or relocate the three §1b production constants.
