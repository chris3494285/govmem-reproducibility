# GovMem JSS Public Release Final Signoff

Date: 2026-11-08  
Scope: current `govmem-jss-public` staging package

## 1. Decision

## **SAFE TO PUBLISH**

The license text and material scope are internally consistent, the public test
suite is green, all manifests verify, and the security scan is clean.

The responsible humans confirmed that GovMem is independent personal research,
that no employer, university, funder, or contractor owns or restricts the
original research materials, that all coauthors approve public distribution,
and that third-party restrictions remain in force. The confirmation is
recorded in `LICENSE_APPROVAL.md`; it is not represented as an independent
legal opinion.

No Git repository was initialized, no commit or push occurred, and no
visibility was changed.

## 2. MIT license and scope

`LICENSE` contains MIT terms and names the four coauthors:

- Bowen Liu
- Youting Wang
- Bonan Shen
- Dingyan Shang

`LICENSE_SCOPE.md` limits the grant to original GovMem implementation and test
code under:

- `govmem/`
- `eval/`
- `baselines/`
- `scripts/`
- `tests/`

The MIT grant does not cover:

- benchmark data or identifiers;
- benchmark-derived aggregate results;
- OAKS-derived material;
- prose documentation or reports;
- configuration files;
- `CITATION.cff`;
- `third_party/**` or any upstream license text.

The permission portion of `LICENSE` is MIT-compatible and does not purport to
relicense third-party material. The root-level placement means repository
hosting interfaces may present MIT as the repository license; `LICENSE_SCOPE.md`
and `third_party/README.md` provide the necessary qualifications.

Coauthor approval, MIT intent, independent-personal-research status, and the
absence of employer/funder/contractor restrictions on the original materials
are recorded in `LICENSE_APPROVAL.md`.

## 3. Documentation, configuration, and aggregate redistribution

`MATERIALS_REDISTRIBUTION_NOTICE.md` records the coauthors' permission to
publicly distribute the original documentation, non-secret configurations, and
aggregate results contained in this package.

This permission:

- covers the original material in this staged package;
- does not apply the MIT grant to that material;
- does not grant rights beyond the upstream terms governing derived data;
- does not cover or relicense third-party benchmark material.

OAKS-derived aggregate files remain limited to noncommercial research and
require the included Adobe Research License. The exact 51-file inventory is in
`OAKS_DERIVED_ARTIFACTS.md`. No OAKS data file was excluded because the license
explicitly permits noncommercial-research derivative redistribution with a
license copy.

## 4. Third-party notices

All required package-level notices are retained:

| Artifact | Status |
|---|---|
| `third_party/licenses/Adobe-Research-License-OAKS-Novel.txt` | Present; SHA-256 verified |
| `third_party/licenses/Apache-2.0-Memora.txt` | Present; SHA-256 verified |
| `third_party/licenses/MIT-LongMemEval.txt` | Present; standard MIT text |
| `third_party/README.md` | Present and scoped to upstream notices |
| `THIRD_PARTY_DATASETS.md` | Present; sources, versions, restrictions, and citations recorded |
| `OAKS_DERIVED_ARTIFACTS.md` | Present; exact OAKS-derived file inventory |

Recorded provenance:

- LongMemEval:
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned,
  revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`,
  `longmemeval_oracle.json`,
  SHA-256 `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`,
  MIT.
- Memora: https://github.com/geniesinc/Memora,
  revision `a6493188efc836d6511ed5e4163fe3ba87da30ff`,
  `data/` corpus, Apache-2.0.
- OAKS Novel: https://github.com/adobe-research/OAKS,
  revision `0bbc4b3877eae5f9335f07c5e485aabf1d9136e9`,
  `data/oaks-novel/oaks-n.split_2k.json`, Adobe Research License.

These are package-recorded identifiers. They were not independently re-queried
online during this offline signoff.

## 5. Security and integrity checks

Checks performed against the complete staged directory:

- credential and API-token patterns;
- PEM private-key blocks;
- `.env` and private filenames;
- absolute user/home paths;
- email addresses;
- benchmark question/answer/context fields;
- request, response, judge, and adjudication records;
- per-example, block, and slot identifiers;
- content-fingerprint fields;
- symlinks, caches, and Git metadata;
- exact OAKS-derived artifact inventory;
- license-copy hashes;
- manifest completeness and all SHA-256 values.

Result: **PASS**. No secret, private record, restricted raw data, local path, or
unexpected identifier was found.

## 6. Tests and manifests

Public offline suite:

```text
525 passed, 230 skipped, 0 failed
```

Private-marked developer mode was also executed from the staging directory:

```text
220 failed, 525 passed, 6 skipped, 4 errors
```

The failures were inspected: 218 were missing-file errors for withheld licensed
or private artifacts, two were assertions on absent legacy/manuscript inputs,
and four were fixture errors for the missing licensed manifest. No test marker
was weakened to produce the public result.

Manifest state:

- 454 payload files;
- three manifest files;
- 457 total files;
- every payload path and SHA-256 verified;
- no unlisted files.

## 7. Human confirmations

The required ownership and distribution confirmations are recorded:

1. **Coauthor approval:** all four coauthors approve the public release.
2. **Independent research:** GovMem is independent personal research.
3. **Employer rights:** no employer, university, funder, or contractor owns or
   restricts the original research materials.
4. **Third-party scope:** existing third-party license restrictions remain in
   force.
5. **Code license:** the intended license for original GovMem code is MIT.

No contributor or employer-right blocker remains after those confirmations.
The following external provenance items remain documented warnings rather than
package redistribution blockers:

- confirm the canonical LongMemEval copyright-holder string when available;
- independently re-query the recorded upstream commits and dataset hashes when
  network access is permitted; and
- preserve the noncommercial-research restriction on all OAKS-derived output.

## 8. Non-blocking provenance warnings

- The OAKS source file is identified by repository revision and filename, but
  no content hash is recorded.
- The complete Memora corpus is not content-pinned by one immutable aggregate
  hash.
- Upstream repository revisions and licenses are recorded in the package but
  were not network-verified during this offline review.
- The OAKS-derived aggregates remain restricted to noncommercial research.

## 9. Reproducibility boundaries

The package directly supports verification of all reported aggregate counts
and effects, including 731 bases, 2,080 pairs, 4,891 conditions, six providers,
298 resolved Formal-300 triplets, 238 resolved base clusters, and the 10,000
resample/seed `20261010` bootstrap.

Full reconstruction still requires:

- the independently obtained licensed datasets;
- the private row-level model outputs;
- private judge routing and adjudication records; and
- the manuscript checkout for manuscript-dependent tests.

The release does not claim full end-to-end reproducibility.

## 10. Final status

**SAFE TO PUBLISH** under the recorded ownership confirmations and retained
third-party restrictions.

Coauthor consent and the code-only MIT grant are recorded. Distribution
permission for original documentation, configuration, and aggregates is
documented separately. Third-party notices, provenance, and usage restrictions
are retained. No technical, privacy, or ownership blocker remains in the staged
package.

Publication must still preserve this package exactly as audited, retain the
noncommercial-research notice for OAKS-derived aggregates, and avoid importing
the original development repository's Git history.
