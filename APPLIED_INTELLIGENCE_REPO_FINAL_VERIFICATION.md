# Applied Intelligence repository final verification

Date: 2026-11-09

## Scope

This follow-up resolves the verification issues left by the Applied
Intelligence repository metadata update. It changes only current-facing
documentation, metadata checks, and checksum declarations.

Unchanged:

- Experimental results and published aggregate statistics.
- Frozen JSON, JSONL, and CSV payloads.
- Provider outputs and scoring records.
- Existing scientific analysis scripts.
- The original 454-entry `MANIFEST.sha256`.
- Git history, branches, remotes, commits, and publication state.

The unchanged audited release commit is:

`b0afd0792df6631096756865eb3713923d17c9a3`

## Changed files

Cumulative tracked modifications from the Applied Intelligence metadata work
and this verification follow-up:

1. `README.md`
2. `CITATION.cff`
3. `ARTIFACTS.md`
4. `docs/JSS_PUBLIC_RESULTS.md`
5. `pyproject.toml`
6. `tests/test_jss_publication_claims.py`

New files:

1. `MANIFEST_APPLIED_INTELLIGENCE.sha256`
2. `APPLIED_INTELLIGENCE_REPO_FINAL_VERIFICATION.md`

The earlier report `APPLIED_INTELLIGENCE_REPO_METADATA_UPDATE.md` remains in
the working tree as the record of the preceding update.

The pre-existing untracked file `JSS_FINAL_EVIDENCE_VERIFIABILITY_AUDIT.md`
was not modified.

## Citation-title test

The assertion in `tests/test_jss_publication_claims.py` read the current
`CITATION.cff` directly. It therefore tested current manuscript metadata, not
frozen historical provenance.

Only the title assertion was changed. It now expects:

`Evaluating Long-Term Memory Reliability in LLM Agents Under Governance-State Faults: A Cross-Model Study`

The author-order, license-scope, materials-license, and DOI-absence assertions
were retained unchanged.

## Historical audit path

The file `results/answer_leakage_audit.json` does not exist:

- in the current working tree;
- at the audited commit `b0afd0792df6631096756865eb3713923d17c9a3`;
- under that path anywhere in reachable Git objects or history.

`ARTIFACTS.md` retains the historical expected SHA-256, count, and audit
record. `README.md` now states that the historical audit output and its
text-bearing v2 input manifest are not distributed in the aggregate-only
release. It also states that local auditing requires rebuilding the v2
manifest from licensed sources.

No missing audit artifact was regenerated, recreated, or invented.

## Manifest design

### Original audited manifest

`MANIFEST.sha256` remains unchanged as the historical integrity declaration
for the original audited release.

- Entries: 454.
- Manifest SHA-256:
  `b52bd4fb18ebba16cd1815761e4d4d0d54e691512bb8ef146bdf42ec60e56cf9`.
- Git diff against `HEAD`: none.

### Applied Intelligence metadata manifest

`MANIFEST_APPLIED_INTELLIGENCE.sha256` is a separate, five-entry manifest for
the current-facing metadata files:

- `README.md`
- `CITATION.cff`
- `ARTIFACTS.md`
- `docs/JSS_PUBLIC_RESULTS.md`
- `pyproject.toml`

It does not include the verification test, reports, experimental payloads, or
the original manifest. It neither covers nor replaces `MANIFEST.sha256`.
README documentation makes this scope explicit.

Metadata manifest SHA-256:

`2bdc92d6e45ab0994dcd4870e0ab8d695aabb0269bbcd0e66aca255842c72469`

## Verification results

### Formatting and citation metadata

- `git diff --check`: PASS.
- `CITATION.cff` YAML parsing with installed PyYAML: PASS.
- Exact Applied Intelligence title: PASS.
- Author order Bowen Liu, Youting Wang, Bonan Shen, Dingyan Shang: PASS.
- Repository URL
  `https://github.com/chris3494285/govmem-reproducibility`: PASS.
- No DOI, volume, issue, or page fields: PASS.
- `cffconvert` remains unavailable, so schema-level CFF validation was not
  run. No dependencies were installed.

### Corrected publication-metadata test

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -o addopts='' -q -p no:cacheprovider tests/test_jss_publication_claims.py::test_license_and_citation_metadata_are_scoped
```

Result:

`1 passed in 0.01s`

### Full offline public suite

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -o addopts='' -q -p no:cacheprovider
```

Result:

`525 passed, 230 skipped in 3.44s`

The skips are the existing public-release exclusions for private, local, or
licensed inputs. No network access, provider API call, model generation, or
semantic judging was performed.

## Separate integrity results

### 1. Historical manifest against the original audited commit

Verification method:

1. Extracted an isolated `git archive` of commit
   `b0afd0792df6631096756865eb3713923d17c9a3`.
2. Ran `shasum -a 256 -c MANIFEST.sha256` in that extraction.

Result:

| Check | Result |
|---|---:|
| Historical entries verified | 454 / 454 |
| Historical entries failed | 0 |
| Command exit | PASS, 0 |

This verifies the original audited release snapshot. It does not describe the
current metadata working tree.

### 2. Current frozen scientific payload integrity

The original manifest was applied to the current working tree after excluding
only the six intentionally changed metadata/verification files:

- `README.md`
- `CITATION.cff`
- `ARTIFACTS.md`
- `docs/JSS_PUBLIC_RESULTS.md`
- `pyproject.toml`
- `tests/test_jss_publication_claims.py`

Result:

| Check | Result |
|---|---:|
| All unchanged original entries | 448 / 448 PASS |
| Original `data/` and `results/` entries | 109 / 109 PASS |
| Working-tree `data/` and `results/` Git diff | none |
| Unfiltered current-tree original manifest | 448 OK, 6 expected metadata/test mismatches |
| `MANIFEST.sha256` Git diff | none |

The six unfiltered mismatches are exactly the intended metadata and test
changes. No additional payload mismatch exists.

### 3. Current Applied Intelligence metadata-release integrity

Command:

```bash
shasum -a 256 -c MANIFEST_APPLIED_INTELLIGENCE.sha256
```

Result:

| Check | Result |
|---|---:|
| Metadata entries | 5 / 5 PASS |
| Metadata entries failed | 0 |
| Original historical manifest replaced | No |
| Experimental payload covered by metadata manifest | No |

## Aggregate and scientific evidence confirmation

The previously verified aggregate values remain unchanged:

- 731 benchmark-grounded bases.
- 2,080 supported base-fault pairs.
- 4,891 condition instances.
- Six generating providers.
- 12,480 provider-replicated pair evaluations.
- 298 resolved native-style triplets.
- 238 distinct resolved bases.

No experimental result, model output, scoring record, aggregate file, or
frozen scientific payload was modified by this follow-up.

## Remaining limitations

1. The full suite retains 230 expected skips because private, local, or
   licensed inputs are not distributed.
2. CFF YAML syntax is validated, but schema-level validation remains
   unavailable because `cffconvert` is not installed.
3. The original 454-entry manifest intentionally does not validate against the
   metadata-updated working tree as a whole. Its frozen content remains valid,
   and the six expected working-tree differences are documented above.
4. The historical `results/answer_leakage_audit.json` artifact is unavailable
   and is not distributed. Only its historical finding, expected hash, and
   reproduction requirements are documented.
5. GitHub repository settings were not inspected or changed because this task
   prohibits API calls. The prior manual description recommendation remains.
6. The pre-existing untracked `JSS_FINAL_EVIDENCE_VERIFIABILITY_AUDIT.md`
   should be excluded from a commit unless its inclusion is separately
   reviewed.

## Readiness assessment

The repository is ready for human review of a documentation-and-verification
commit.

The corrected test passes, the metadata manifest validates, all current frozen
`data/` and `results/` payload hashes pass, and the original audited manifest
remains byte-identical to `HEAD`. No experimental content changed.

No commit or push was performed. All changes remain available in the working
tree for review.
