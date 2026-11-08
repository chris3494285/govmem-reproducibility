# Final license and release report — GovMem aggregate-only package

Date: 2026-10-14. This report closes out the third-party licensing and
attribution work for the GovMem aggregate-only public release and states
whether the package is ready to be pushed to an independent, fresh-history Git
repository.

Constraints honoured: no paid APIs, no experiment reruns, no changes to paper
results or frozen data, no commit, no push. Every step below is offline and
read-only with respect to frozen artifacts.

## 1. License verification against the exact versions used

Each license was checked against the specific upstream artifact GovMem used,
not against a general statement about the project.

| Dataset | Upstream artifact used | Version evidence | License | Verified from |
|---|---|---|---|---|
| LongMemEval (cleaned) | `longmemeval_oracle.json` from `xiaowu0162/longmemeval-cleaned` | content hash `821a2034…38620c`, recorded in `data/final_main_governance_condition_manifest_v3.freeze.json`; hash re-verified against the local copy during this audit and it matches | **MIT** (dataset-card `license: mit`) | the dataset card front-matter in the upstream repository |
| Memora | `data/` corpus from `geniesinc/Memora` @ `a6493188` | repo commit + the derived-source hash `c55f625e…f35cfc8` in `sampling_lock.json` | **Apache-2.0** | the repo `LICENSE` (verbatim, SHA-256 `c71d239d…5d0ab4`) and the explicit dataset statement in `data/README.md` |
| OAKS Novel | `data/oaks-novel/oaks-n.split_2k.json` from `adobe-research/OAKS` @ `0bbc4b38` | repo commit + filename | **Adobe Research License** | the repo `LICENSE` (verbatim, SHA-256 `d9a8d2d8…751739`) |

**Result: all three licenses are verified for the intended use.** No dataset's
terms contradict the aggregate-only scope. Two dataset-specific findings are
recorded below.

### Dataset-specific findings

- **OAKS Novel — noncommercial-only, copy-of-license required.** The Adobe
  Research License grants redistribution of the Licensed Materials and of
  derivative works *for noncommercial research purposes only*, and requires
  that recipients of a redistribution receive a copy of the License (§1.1 B,
  §1.3, §2). This release therefore (a) ships the verbatim Adobe license, and
  (b) is offered for noncommercial research use only. This is the binding
  restriction on the package.
- **LongMemEval — MIT declared at card level only.** The license is stated as
  metadata (`license: mit`) in the dataset card; no `LICENSE` file and no
  verbatim copyright-holder string are published upstream. The notice shipped
  here uses the standard MIT text with a descriptive holder line, flagged for
  confirmation.
- **Memora — dataset explicitly Apache-2.0, no upstream NOTICE.** `data/README.md`
  restates Apache-2.0 for the corpus, so the license covers the data, not just
  the code. The repository ships no `NOTICE` file, so there is no upstream
  NOTICE text that must be propagated.
- **OAKS-B was not used.** Only the OAKS-Novel split was used, so BABILong's
  terms (which apply to the OAKS-B synthetic split) do not enter this release.

## 2. Attribution and notice compliance

Shipped in this release:

| File | Purpose |
|---|---|
| `THIRD_PARTY_DATASETS.md` | per-dataset source, license, version evidence, restrictions, citations, obligations |
| `third_party/README.md` | notice index; states these are upstream licenses, not a GovMem license |
| `third_party/licenses/Apache-2.0-Memora.txt` | verbatim Apache-2.0 (identical SHA-256 to upstream) |
| `third_party/licenses/Adobe-Research-License-OAKS-Novel.txt` | verbatim Adobe Research License (identical SHA-256 to upstream) |
| `third_party/licenses/MIT-LongMemEval.txt` | standard MIT text + confirmation note |

Obligation-by-obligation:

- **Include license text** — done for all three.
- **Retain copyright/attribution notices** — recorded in `THIRD_PARTY_DATASETS.md`.
- **Give recipients a copy of the Adobe license** — satisfied by shipping it and
  indexing it from `third_party/README.md`.
- **Cite the datasets** — BibTeX for all three is in `THIRD_PARTY_DATASETS.md`.
- **State changes (Apache-2.0 §4(b))** — not triggered: GovMem redistributes no
  Memora file; derived manifests are new works described in `ARTIFACTS.md`.

No third-party file was modified, and nothing here relicenses third-party
material.

## 3. Separation of GovMem's own licensing

- **No repository-wide license was assigned.** No `LICENSE`, `LICENSE-*`, or
  `CITATION.cff` file was added, and no license field was added to
  `pyproject.toml`. Choosing a redistribution license is a maintainer decision.
- The third-party notices live under `third_party/` and are explicitly scoped
  to the upstream datasets; they grant no rights to GovMem's own code or
  results.
- Consequence to be aware of: absent a chosen license, GovMem's own material
  defaults to "all rights reserved". Publishing the repository is still
  possible; downstream reuse of GovMem's code is not granted until the
  maintainers choose a license.

## 4. Aggregate-only scope preserved

Independently re-checked in the rebuilt candidate:

| Requirement | Status |
|---|---|
| No benchmark question/answer/context text | satisfied — 0 hits |
| No per-example benchmark identifiers | satisfied — 0 hits in data artifacts |
| No per-row content fingerprints | satisfied — 0 hits |
| No row-level score files | satisfied — none present |
| No private artifacts / private mapping | satisfied — none present |
| No `.git` history | satisfied — no `.git` directory |

The only benchmark-derived tokens anywhere in the package are the three
documented, fail-closed production constants recorded in
`FINAL_GITHUB_RELEASE_AUDIT.md` §1b.

## 5. Rebuild and verification

```
/tmp/govmem-aggregate-only-release-v3
```

| Check | Result |
|---|---|
| Build from the explicit allowlist | **PASS** — 406 files, 0 violations |
| Structure + content audit | **PASS** — 406 files on disk, 406 manifest entries, 0 violations |
| SHA-256 (`shasum -a 256 -c MANIFEST.sha256`) | **PASS** — 406/406 OK, 0 non-OK |
| Public offline suite (candidate) | **PASS** — 568 passed, 253 skipped, 0 failed |
| Public offline suite (developer tree) | **PASS** — 787 passed, 293 skipped, 0 failed |
| `.git` / private mapping in candidate | absent / absent |

Build command:

```bash
python scripts/build_public_release_candidate.py \
  --repo-root . \
  --allowlist release/aggregate_only_release_allowlist.json \
  --output /tmp/govmem-aggregate-only-release-v3 --force
```

## 6. Outstanding issues

1. **Noncommercial-research restriction (binding, not a defect).** The package
   includes OAKS-Novel-derived aggregates, so it may be redistributed only for
   noncommercial research purposes. This is stated in `README.md`,
   `THIRD_PARTY_DATASETS.md`, and `third_party/README.md`. If the maintainers
   want a permissively reusable release, the OAKS-Novel-derived aggregates must
   be removed or separately cleared.
2. **OAKS-Novel revision not hash-pinned.** The exact upstream revision is
   identified only by filename and repository commit. Recommended: record the
   file's SHA-256 in `THIRD_PARTY_DATASETS.md`.
3. **LongMemEval copyright-holder string unconfirmed.** The card declares MIT
   but publishes no holder string; the shipped MIT notice carries a descriptive
   line that should be confirmed.
4. **GovMem's own license unassigned.** Maintainer decision. Until then, the
   default "all rights reserved" applies to GovMem's own material.
5. **Formal-300 297-vs-298 provenance note** (carried over from the previous
   audit; documentation only, no statistic changed).

None of these requires a change to experimental data, and none of them is a
technical defect in the package.

## 7. Readiness determination

**The aggregate-only package is READY to be pushed to an independent,
fresh-history Git repository**, subject to the two statements that already
appear in the shipped documentation:

- the package is for noncommercial research use only (because it carries
  OAKS-Novel-derived aggregates), and
- GovMem's own redistribution license is not yet assigned.

Recommended (not blocking) before publishing: confirm the LongMemEval copyright
holder string, pin the OAKS-Novel source hash, and add the Formal-300
provenance note.

Verdict in one line: **licenses verified, attribution shipped, scope preserved,
verification green — publish to a fresh-history repository under a
noncommercial-research notice and an unassigned GovMem license.**
