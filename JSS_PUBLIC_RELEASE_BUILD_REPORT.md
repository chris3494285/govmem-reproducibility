# GovMem JSS Public Release Build Report

> **Historical pre-remediation report.** This report records the initial
> staging audit. License, citation, and provenance changes are superseded by
> `JSS_PUBLIC_RELEASE_REMEDIATION_REPORT.md`.

Build date: 2026-11-08  
Staging directory: `$USER_HOME/Devp/govmem-jss-public`  
Source worktree: `<PUBLIC_REPO_ROOT>`

## 1. Final decision

## **NOT SAFE TO PUBLISH**

The staged package is technically clean, complete enough to verify the
manuscript's aggregate results, and passes the public offline test suite. No
secrets, benchmark text, row-level results, model responses, judge records,
human adjudications, or machine-specific paths were found in the staging
directory.

Publication is withheld because licensing decisions remain unresolved:

1. GovMem's own code, tests, documentation, and aggregate results have no
   selected redistribution license.
2. OAKS Novel-derived aggregates remain subject to the Adobe noncommercial
   research restriction, and the maintainers have not formally approved that
   interpretation for this release.
3. The LongMemEval MIT attribution does not have a confirmed copyright-holder
   string.
4. `CITATION.cff` is absent.

No GitHub repository was created or published. No Git history was initialized
or copied. No visibility was changed and nothing was pushed.

## 2. Build method and exact contents

The staging directory was produced only through:

```bash
python3 scripts/build_public_release_candidate.py \
  --repo-root . \
  --allowlist release/aggregate_only_release_allowlist.json \
  --output $USER_HOME/Devp/govmem-jss-public \
  --force
```

The build is deny-by-default. It selects only paths matched by the explicit
include rules, subtracts reviewed exclusions, applies path sanitization and
reviewed JSON-field redaction, and fails before writing if any forbidden
content remains. Symbolic links and output inside the source repository are
rejected.

The final package has:

- **446 payload files** listed in `FILE_MANIFEST.json`;
- **449 total files on disk**, including `MANIFEST.sha256`,
  `FILE_MANIFEST.json`, and `FILE_MANIFEST.csv`;
- no `.git` directory or `.git` file;
- 25 path-sanitized files;
- four JSON/JSONL files with one reviewed field redacted each;
- zero allowlist content violations.

The exact path list for all 446 payload files is the sorted `files[].path`
array in `FILE_MANIFEST.json`. Its stable inventory identity is:

```text
SHA-256 of the newline-delimited sorted payload paths:
8616c502bd29812572ddbe0c25288e6e4308d9f64370be8e5b2fd0396ac9056a
```

`MANIFEST.sha256` independently records the SHA-256 of every payload file.
The source-level exact selection is
`release/aggregate_only_release_allowlist.json`, which is itself included in
the package.

### Contents by category

| Category | Files | Classification |
|---|---:|---|
| Package source (`govmem/`, `eval/`, `baselines/`) | 60 | Original GovMem code |
| Analysis and experimental scripts | 126 | Original GovMem code |
| Tests and author-created fixtures | 108 | Original test code; synthetic fixtures only |
| Configurations | 15 | Original non-secret experiment configuration |
| Documentation and root reports | 24 | Original GovMem documentation |
| Final v3 aggregates | 46 | Original derived aggregate results |
| Formal-300 aggregates and protocol | 11 | Original derived aggregate results |
| B4 aggregate validation | 18 | Original derived aggregate results |
| Clean-repeat aggregate | 7 | Original derived aggregate results |
| Other aggregate results | 24 | Original derived aggregate results |
| Frozen metadata | 3 | Original hash/count metadata |
| Third-party notices | 4 | One index plus three upstream license texts |

The included source tree contains all of the following:

- RQ1-RQ4 implementation and analysis entry points;
- governance schemas, fault operators, enforcement, and context projection;
- frozen analysis/failure specifications and provider budget configurations;
- Formal-300 protocol, aggregate results, and statistical implementations;
- RQ1 Observability V2 code and aggregate tables;
- native paired block and base-cluster bootstrap implementations;
- scoring-consistency and common-score sensitivity code and aggregate output;
- public aggregate statistics and manuscript-count verification;
- test coverage for the release builder, public aggregate construction,
  manuscript counts, and security behavior.

## 3. Builder and fingerprint remediation

The previous builder failure was caused by two literal `pilot-block` values in
`tests/test_jss_rq1_observability.py`. Those values are deterministic digests of
frozen matched-block identities, not benchmark text and not reversibly
recoverable. They nevertheless identify specific source-derived units, so the
aggregate-only policy correctly treated them as restricted release material.

The remediation does not add an exemption for those identifiers:

- the hard-coded values were removed from the public test;
- the test now derives the two unresolved blocks from the private frozen
  metrics and observations at runtime;
- the exact count and clean/corrupted/enforced join are still asserted;
- the integration test remains marked as requiring private artifacts;
- builder failures print only the path, line, and rule, never the matched
  content;
- a new regression test proves that the builder fails closed on a
  fingerprint-shaped value without echoing it;
- frozen Formal-300 block identity constants were removed from public code or
  replaced by private environment-supplied identities;
- the frozen block fingerprint is removed from public JSON and protocol text.

The security checks were strengthened, not weakened. The final allowlist audit
still rejects source-derived identifiers, block/slot fingerprints, raw
benchmark text fields, request identifiers, content fingerprints, credentials,
absolute user paths, and email addresses.

## 4. Validation results

### Public offline suite

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -m pytest -o addopts='' -q -p no:cacheprovider
```

Result:

```text
523 passed, 230 skipped, 0 failed
```

Skipped tests are intentional private-input coverage:

- 162 tests require licensed benchmark data;
- 66 tests require private/local experiment artifacts;
- the remaining two skipped cases report unavailable optional artifacts at
  runtime.

The formal-300 integration test module and the separate B4 full-native canary
subsystem are outside the aggregate package. Their source/tests were excluded
when they depended on private plans, private ledgers, or source-derived
recovery identifiers.

### Manifest and content validation

```text
files on disk: 446
manifest entries: 446
content violations: 0
PASS: candidate is internally consistent and free of forbidden content
```

`shasum -a 256 -c MANIFEST.sha256` verified all 446 payload files.

### Independent publication audit

```text
JSS_TECHNICAL_AUDIT_STATUS=PASS
JSS_LICENSE_STATUS=BLOCKED (GovMem redistribution license not selected)
```

The independent audit also verified:

- no `.git` directory;
- no symbolic links;
- no raw benchmark question/answer/context fields;
- no row or block identifiers;
- no request, response, judge, payload, ledger, or adjudication files;
- no secret-shaped strings or private-key blocks;
- no absolute `/Users` or `/home` paths;
- no private file names;
- exact Apache-2.0 and Adobe license-copy hashes.

The only question/answer-shaped data are the author-created synthetic fixtures
under `tests/fixtures/memora-mini/`. They contain invented test content and no
LongMemEval, Memora, or OAKS Novel identifier or text.

Private material cannot enter through the current build path: only explicit
allowlist matches are read, private subtrees and row-level result paths are
excluded, symbolic links are rejected, and the output is audited before
publication. The staging copy contains no Git history that could bypass that
selection.

### Private-input fail-closed behavior

Running the scoring-consistency analysis from the public staging directory
without private inputs terminates with an explicit missing-manifest error and
writes no result. Running the public-aggregate builder without the local
non-public RQ1 summary terminates with an explicit missing-input error.

This is intentional. The aggregate outputs are shipped and verifiable, but
they cannot be recomputed from the package alone.

## 5. Manuscript consistency

The released aggregates were checked against the final manuscript values:

| Claim | Released value | Result |
|---|---:|---|
| Benchmark-grounded bases | 731 | MATCH |
| Supported base-fault pairs | 2,080 | MATCH |
| Conditions | 4,891 | MATCH |
| Generating providers | 6 | MATCH |
| Provider-replicated pair evaluations | 12,480 | MATCH |
| Resolved Formal-300 triplets | 298 | MATCH |
| Resolved Formal-300 distinct bases | 238 | MATCH |
| Formal base-cluster bootstrap resamples | 10,000 | MATCH |
| Formal base-cluster bootstrap seed | 20261010 | MATCH |
| Identical-response triplets | 64 | MATCH |
| Identical-response/score-change cases | 1 | MATCH |

The final manuscript's principal aggregate effects also match:

- control-plane projection change: 12,012 / 12,480 = 96.25%;
- projection restoration: 12,480 / 12,480 = 100%;
- Formal-300 corruption: +35.54 pp, CI [29.78, 41.12];
- Formal-300 recovery: +32.67 pp, CI [26.94, 38.36];
- scoring-sensitivity change to LongMemEval damage: 0.68 pp;
- scoring-sensitivity change to pooled damage: 0.34 pp.

The public non-adjudicated Formal-300 analysis resolves 297 triplets. The
package labels it separately from the final 298-triplet human-adjudicated
analysis.

## 6. Licensing classification

### Original GovMem material

Classification: **original code/results; redistribution license unresolved**.

This applies to package code, tests, configurations, documentation, and
aggregate results created by the GovMem authors. No blanket license was added,
because the repository does not establish the maintainers' chosen license.
Until one is selected, the default all-rights-reserved position applies.

### Third-party license copies

| Path | Classification | Finding |
|---|---|---|
| `third_party/licenses/Apache-2.0-Memora.txt` | Third-party, verified | Hash matches the documented upstream Apache-2.0 license text. |
| `third_party/licenses/Adobe-Research-License-OAKS-Novel.txt` | Third-party, verified | Hash matches the documented upstream Adobe license text. |
| `third_party/licenses/MIT-LongMemEval.txt` | Third-party, attribution unresolved | Standard MIT text is included, but the upstream dataset card does not provide a verbatim copyright-holder string. |
| `third_party/README.md` | Original GovMem notice index | Contains no grant for GovMem's own material. |

### Benchmark-derived aggregates

Classification: **original derived aggregates with upstream restrictions**.

- LongMemEval is documented as MIT.
- Memora is documented as Apache-2.0.
- OAKS Novel uses the Adobe Research License, which permits covered
  redistribution only for noncommercial research and requires a copy of the
  license.

The package is therefore framed as noncommercial academic research material.
Whether every group-level statistic is a derivative work remains a legal
decision. The package does not assert broader rights.

### Restricted material excluded

Classification: **restricted or unresolved; excluded**.

- raw LongMemEval, Memora, and OAKS Novel data;
- condition and base manifests containing benchmark text;
- row-level provider scores and per-instance/per-base tables;
- model responses, parsed answers, and request identifiers;
- prompts, semantic payloads, response hashes, and content fingerprints;
- judge requests, judge outputs, repair ledgers, and execution plans;
- human-adjudication records and recovery authorizations;
- private canary and synthetic execution states;
- internal manuscript review files and local audit drafts;
- local paths, caches, credentials, and `.env` files.

## 7. Reproducibility coverage

Directly verifiable from the package:

- all manuscript headline counts;
- final v3 provider/source/fault/event aggregates;
- Formal-300 aggregate estimates and interval methods;
- RQ1 Observability V2 aggregate results;
- scoring-consistency counts and common-score sensitivity;
- license texts, dataset acquisition instructions, and limitation statements;
- hashes and manifest completeness.

Requires independently obtained licensed data:

- rebuilding the base and condition manifests;
- checking source-derived eligibility and context hashes;
- regenerating row-level model inputs.

Requires private experiment records:

- recomputing row-level provider scores;
- recomputing the final 298/898 Formal-300 analysis;
- reproducing the private judge-routing and human-adjudication trace;
- recomputing the scoring inconsistency from answer/prompt identities.

The package does not claim full end-to-end reproducibility.

## 8. Remaining blockers

1. Select and add a GovMem-owned license that does not imply a license for
   third-party benchmark material.
2. Obtain maintainer and institutional approval for distributing
   OAKS-derived aggregate statistics under the Adobe noncommercial research
   terms, or remove those aggregates and revise the manuscript's claims.
3. Confirm the LongMemEval copyright-holder string and update the MIT notice if
   the upstream authors provide a canonical notice.
4. Add `CITATION.cff` with the final JSS authorship and repository metadata.
5. Pin the exact OAKS Novel and Memora source revisions/files used where
   feasible. LongMemEval is already content-hash pinned.
6. After licensing and citation edits, rebuild the candidate, rerun the full
   public suite and independent audit, and verify the manifest again.

These are publication gates. They are not privacy or technical-release defects
in the staged files.

## 9. Final publication status

**NOT SAFE TO PUBLISH until the licensing and citation blockers above are
resolved.**

The staging directory itself is safe to retain, inspect, transfer privately,
and use for the remaining licensing review. It must not be made public or
pushed to a public repository in its current state.
