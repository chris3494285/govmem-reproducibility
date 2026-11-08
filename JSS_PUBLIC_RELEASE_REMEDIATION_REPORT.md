# GovMem JSS Public Release Remediation Report

> **Historical remediation record.** Coauthor approval, final license-scope
> verification, and the current release decision are superseded by
> `JSS_PUBLIC_RELEASE_FINAL_SIGNOFF.md`.

Date: 2026-11-08  
Scope: current `govmem-jss-public` staging package only

## 1. Final decision

## **NOT SAFE TO PUBLISH**

The package is technically clean, its public tests pass, and every staged file
has a documented copyright/licensing category. Publication is still blocked
because no offline record establishes that all four authors own the original
material and approve the proposed code-only MIT grant.

No Git history was initialized, no commit or push occurred, no GitHub
visibility was changed, and no external service or LLM was called.

## 2. Files added or changed

### Added

- `LICENSE`: proposed MIT grant naming Bowen Liu, Youting Wang, Bonan Shen, and
  Dingyan Shang. It is explicitly limited to original GovMem implementation
  and test code.
- `LICENSE_SCOPE.md`: exact license coverage, exclusions, and the pending
  ownership/approval gate.
- `CITATION.cff`: CFF 1.2.0 metadata using the final manuscript title and the
  requested four-author order. It contains no DOI, journal title, or fabricated
  journal publication date.
- `OAKS_DERIVED_ARTIFACTS.md`: exact inventory of all 51 staged `data/` and
  `results/` files containing OAKS-derived labels or aggregate values.
- `JSS_PUBLIC_RELEASE_REMEDIATION_REPORT.md`: this report.

### Changed

- `README.md`: replaced the former no-license statement with the code-only MIT
  scope, OAKS artifact inventory, and citation metadata.
- `THIRD_PARTY_DATASETS.md`: added immutable repository references, recorded
  offline-verification limits, documented the OAKS redistribution basis, and
  separated remaining provenance gaps from unresolved third-party rights.
- `docs/OBTAINING_DATASETS.md`: added official URLs, repository-recorded
  revisions, versions, licenses, restrictions, and OAKS acquisition guidance.
- `third_party/README.md`: clarified that the root MIT file does not relicense
  third-party notices or benchmark-derived material.
- `scripts/audit_jss_publication.py`: now requires the license, scope, citation,
  OAKS inventory, manuscript aggregates, and copyright approval gate.
- `tests/test_jss_publication_claims.py`: added license-scope, citation, and
  exact OAKS-inventory checks.
- `JSS_PUBLIC_RELEASE_BUILD_REPORT.md`: marked as the historical pre-remediation
  report and linked this report as the current status.
- `release/aggregate_only_release_allowlist.json`: added the reviewed license,
  citation, provenance, and remediation files.
- `MANIFEST.sha256`, `FILE_MANIFEST.json`, and `FILE_MANIFEST.csv`: regenerated
  after all remediation changes.

### Excluded

No additional files were excluded in this remediation. The previously excluded
raw benchmark data, row-level results, model responses, judge records,
adjudications, private execution state, manuscript review files, and local
paths remain excluded.

No frozen experiment result was modified.

## 3. GovMem license decision

`LICENSE` is a proposed MIT license for original GovMem implementation and test
code only. Copyright holders are listed as:

- Bowen Liu
- Youting Wang
- Bonan Shen
- Dingyan Shang

The grant does not apply to:

- `third_party/**`;
- upstream dataset files or license texts;
- benchmark question, answer, context, identifier, or fingerprint data;
- benchmark-derived aggregate results, including OAKS-derived aggregates;
- prose documentation and reports unless separately approved; or
- citation and bibliography metadata.

### Ownership and approval status

**UNRESOLVED.** The staged package contains no signed ownership statement,
contributor license agreement, copyright assignment, or explicit approval from
all four authors for this exact MIT text. The user's requested author list does
not by itself establish ownership of every contributed file or authority to
license another contributor's work.

An author-approved record must identify the covered paths, confirm the
copyright holders, and approve the code-only MIT grant before this blocker can
be cleared.

## 4. OAKS licensing decision

### Applicable upstream terms

- Repository: https://github.com/adobe-research/OAKS
- Repository-recorded revision:
  `0bbc4b3877eae5f9335f07c5e485aabf1d9136e9`
- Version used: `data/oaks-novel/oaks-n.split_2k.json`
- License: Adobe Research License
- Included license SHA-256:
  `d9a8d2d8348caaab20299ef6441854cce1312dc29f67923a94f2804826751739`

The included license grants redistribution of the licensed materials,
modifications, and derivative works for noncommercial research purposes,
provided recipients receive a copy of the license. Academic JSS research falls
within the stated noncommercial research definition. The exact upstream data
file hash is not recorded in the package.

### Staged OAKS material

The release contains 51 OAKS-derived metadata, documentation, and aggregate
files. They are enumerated exactly in `OAKS_DERIVED_ARTIFACTS.md`. No raw OAKS
question, answer, context, or per-example record is present.

The release conservatively treats those files as derivative works and relies on
Adobe Research License §1.1(B). The package therefore:

- remains noncommercial-research-only;
- includes the complete Adobe license;
- retains ownership and attribution statements; and
- does not apply the GovMem MIT grant to any OAKS-derived file.

**Decision:** retain the OAKS-derived aggregate files under the Adobe
noncommercial-research restriction. No exclusion is required by the license
text, and the manuscript's published aggregate results remain available.

Original implementation, documentation, and tests that merely refer to OAKS by
name are not listed as OAKS data artifacts.

## 5. LongMemEval and Memora provenance

### LongMemEval (cleaned)

- Repository: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- Repository-recorded revision:
  `98d7416c24c778c2fee6e6f3006e7a073259d48f`
- Version: `longmemeval_oracle.json`
- Recorded file SHA-256:
  `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`
- License: MIT, as declared in the dataset card
- Restriction: retain the copyright and permission notice with substantial
  copies
- Remaining issue: the upstream holder string is not verbatim confirmed

The repository revision and file hash are package-recorded provenance. They
were not independently re-queried online during this offline remediation.

### Memora

- Repository: https://github.com/geniesinc/Memora
- Repository-recorded revision:
  `a6493188efc836d6511ed5e4163fe3ba87da30ff`
- Version: `data/` corpus
- License: Apache License 2.0
- Included license SHA-256:
  `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`
- Restriction: retain notices; state modifications where required
- Remaining issue: no full-corpus immutable file hash is recorded

No raw Memora file is redistributed. The release includes original aggregate
derivatives and the Apache license copy.

## 6. Citation metadata

`CITATION.cff` uses:

- Title: **Stress-Testing the Control Plane of Agent Memory**
- Authors, in manuscript order: Bowen Liu, Youting Wang, Bonan Shen, Dingyan
  Shang
- Type: software
- Version: 0.1.0
- Release date: 2026-11-08

No DOI, journal volume, publication date, or other unpublished bibliographic
identifier was invented. The YAML parses successfully and the title, author
order, and absence of a DOI were checked offline.

## 7. Validation

### Public test suite

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -m pytest -o addopts='' -q -p no:cacheprovider
```

Result:

```text
525 passed, 230 skipped
```

No public test failed.

### Developer/private-input suite

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -m pytest -o addopts='' -q --include-private --tb=line -rfE \
  -p no:cacheprovider
```

Result:

```text
220 failed, 525 passed, 6 skipped, 4 errors
```

This is not a clean developer-suite result. The staged package intentionally
does not contain the licensed manifests, private row-level artifacts, or
external manuscript required by those tests. The failures were inspected
rather than hidden by weakening markers:

- 218 failures were `FileNotFoundError` at missing benchmark/private inputs;
- two assertions depended on the absent legacy dataset and manuscript text;
- the four errors were fixture setup errors for the missing licensed manifest.

The private/developer suite can only complete in a separate environment with
the licensed datasets, original experiment outputs, and manuscript checkout.

### Manifest integrity

The regenerated package contains 451 payload files plus three manifest files.
All payload files are listed exactly once and all SHA-256 values verify.

### Sensitive-content scan

The final scan checks:

- credential and token shapes;
- PEM private keys;
- absolute user/home paths;
- email addresses;
- raw benchmark text fields;
- request and response identities;
- per-example, block, and slot identifiers;
- private directories and filenames;
- content-fingerprint fields;
- symlinks, caches, and Git metadata.

All checks pass with zero findings in the staged package.

## 8. Remaining licensing and provenance blockers

1. **BLOCKER: owner approval.** All four authors must attest ownership and
   approve this exact code-only MIT grant before `LICENSE` is finalized.
2. **BLOCKER: license coverage of non-MIT original material.** Documentation,
   configurations, and aggregate results are intentionally outside the MIT
   grant. Their public redistribution requires author ownership approval or a
   separately scoped license/notice.
3. **WARNING: LongMemEval holder string.** The MIT license text uses a
   descriptive holder because no canonical holder string is available.
4. **WARNING: OAKS source-file hash.** The upstream repository commit and
   filename are recorded, but the exact OAKS file hash is unavailable.
5. **WARNING: Memora corpus hash.** The commit is recorded, but the complete
   data corpus is not content-pinned.
6. **WARNING: online provenance re-verification.** Network access was not used;
   upstream commits and license hashes are package-recorded rather than
   freshly re-queried.

The OAKS redistribution right itself is adequately documented for this
noncommercial research package, provided the Adobe notice remains included.

## 9. Reproducibility limitations

Directly verifiable from the package:

- 731 bases, 2,080 supported pairs, 4,891 conditions, and six providers;
- 298 resolved Formal-300 triplets over 238 distinct bases;
- 10,000-resample base-cluster bootstrap with seed `20261010`;
- RQ1 Observability V2 aggregate results;
- the isolated scoring inconsistency and its sensitivity;
- all headline aggregate effects and the OAKS artifact inventory.

Requires licensed external data:

- rebuilding source manifests;
- reproducing per-example source eligibility;
- regenerating benchmark-derived inputs.

Requires private experiment records:

- recomputing row-level provider scores;
- recomputing the final human-adjudicated 298/898 Formal-300 result;
- reproducing judge routing, private adjudication, and the low-level
  scoring-consistency trace.

There is no full end-to-end reproducibility claim.

## 10. Publication status

**NOT SAFE TO PUBLISH.**

OAKS redistribution is resolved for noncommercial research in this package, the
citation metadata is valid, and all technical/security checks pass. Publication
must remain blocked until copyright ownership and author approval for the
GovMem license and the non-MIT original material are documented.
