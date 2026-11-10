# Applied Intelligence repository metadata update

Date: 2026-11-09

## Scope

This update changes current-facing documentation, repository metadata, and
citation metadata only. It does not modify experiments, research results,
model outputs, frozen artifacts, generating scripts, tests, manifests, Git
history, branches, or remotes.

The audited release commit remains:

`b0afd0792df6631096756865eb3713923d17c9a3`

The tracked worktree changes were left uncommitted for human review.

## Exact files modified

Tracked modifications:

1. `README.md`
2. `CITATION.cff`
3. `ARTIFACTS.md`
4. `docs/JSS_PUBLIC_RESULTS.md`
5. `pyproject.toml`

Created report:

- `APPLIED_INTELLIGENCE_REPO_METADATA_UPDATE.md`

Pre-existing untracked file preserved without modification:

- `JSS_FINAL_EVIDENCE_VERIFIABILITY_AUDIT.md`

## README changes

- Changed the title to
  `# GovMem: Long-Term Memory Reliability in LLM Agents`.
- Identified the manuscript as the Applied Intelligence submission
  “Evaluating Long-Term Memory Reliability in LLM Agents Under Governance-State
  Faults: A Cross-Model Study.”
- Stated the actual experimental boundary: source-supported governance-state
  fault specifications; the intervention changes the operative model-visible
  context projection; the archived internal governance-state record is not
  independently corrupted; and oracle-backed restoration is a reference
  condition, not an autonomous repair algorithm.
- Replaced the old canonical GitHub URL with
  `https://github.com/chris3494285/govmem-reproducibility`.
- Replaced current-facing “JSS” descriptions with “public GovMem
  reproducibility release,” “final GovMem experimental evidence,” and
  “private-input GovMem analyses.”
- Preserved the technical sections, commands, experiment descriptions,
  aggregate-only scope, historical v2 answer-leakage issue, corrected v3
  results, licensing restrictions, and reproduction limitations.

## CITATION.cff changes

- Set the title to
  `Evaluating Long-Term Memory Reliability in LLM Agents Under Governance-State Faults: A Cross-Model Study`.
- Preserved the author order as Bowen Liu, Youting Wang, Bonan Shen, and
  Dingyan Shang.
- Recorded Bowen Liu as the corresponding author in the abstract text.
- Added the canonical repository URL through `repository-code`.
- Stated that the manuscript is under preparation/submission to Applied
  Intelligence.
- Stated explicitly that the citation does not claim journal publication or
  acceptance.
- Added no DOI, volume, issue, or page fields.
- Preserved the existing software version and `date-released` metadata; the
  latter is the software release date, not a manuscript publication date.
- No `preferred-citation` block exists, so none was created.
- No ORCID or other author identifiers were present to preserve.

## Other documentation updates

- `ARTIFACTS.md`: renamed the current-facing “JSS final-evidence additions”
  heading to “Final-evidence additions” and “JSS interpretation and
  limitations” to “Public interpretation and limitations.” Result paths and
  numerical claims were not changed.
- `docs/JSS_PUBLIC_RESULTS.md`: changed the heading and introductory wording
  from JSS-specific language to “Public results and reproduction scope” and
  “public GovMem reproducibility package.” The filename was retained.
- `pyproject.toml`: replaced the superseded paper title in the package
  description with the journal-neutral repository description.

## Historical JSS references intentionally preserved

These are category B references: historical audit material, frozen artifact
names, actual script/test names, or repository lineage. They were not renamed
or bulk-replaced.

- `docs/JSS_PUBLIC_RESULTS.md`
- `results/jss_public_aggregate_v1/`
- `results/jss_rq1_observability_v2/`
- `scripts/analyze_jss_rq1_observability.py`
- `scripts/audit_jss_rq1_observability.py`
- `scripts/analyze_jss_scoring_consistency.py`
- `scripts/build_jss_public_aggregate.py`
- `scripts/audit_jss_publication.py`
- `tests/test_build_jss_public_aggregate.py`
- `tests/test_jss_publication_claims.py`
- `JSS_PUBLIC_RELEASE_BUILD_REPORT.md`
- `JSS_PUBLIC_RELEASE_REMEDIATION_REPORT.md`
- `JSS_PUBLIC_RELEASE_FINAL_SIGNOFF.md`
- `JSS_FINAL_EVIDENCE_VERIFIABILITY_AUDIT.md` (pre-existing untracked file)

The historical remediation report also retains the formerly used manuscript
title because it records the state of an earlier release. Historical release
reports retain the local staging path `$USER_HOME/Devp/govmem-jss-public` where
it identifies the package that was actually built and audited.

No current-facing tracked documentation now identifies the manuscript as a JSS
submission. The tracked-file searches found no remaining instance of
`govmem-jss-reproducibility`, `JSS paper`, `JSS submission`, or
`Journal of Systems and Software` in current-facing text.

## Remaining old URL occurrences

None in tracked files. Historical reports contain the local staging directory
name `govmem-jss-public`, but not the old canonical GitHub repository URL.
Those directory references were preserved because they record the actual
historical build location.

## GitHub repository description recommendation

The live GitHub API reports the current description as:

`Reproducibility artifacts for the GovMem JSS paper`

No local tracked file controls the GitHub short description. The following
replacement should be entered manually in GitHub repository Settings:

`Reproducibility artifacts for evaluating long-term memory reliability in LLM agents under governance-state faults.`

No GitHub description or other remote setting was modified.

## Verification results

### Formatting

- `git diff --check`: PASS.
- All modified files are text files; no unexpected binary files were changed.

### CITATION.cff

- YAML parse using the installed PyYAML library: PASS.
- `cff-version`: `1.2.0`.
- Exact manuscript title: PASS.
- Author order: PASS.
- Canonical `repository-code`: PASS.
- DOI, volume, issue, and page fields absent: PASS.
- `cffconvert` is not installed, so schema-level CFF validation was not run.
  No dependencies were installed.

### Repository paths

- All currently shipped paths checked in the updated README, artifact map, and
  results/reproduction documentation exist.
- A documented-reference scan examined 82 explicit path-like tokens. Of
  these, 66 resolve in the working tree and 16 refer to intentionally absent
  local, private, licensed, or row-level inputs/outputs.
- The intentionally absent references include the row-level score
  directories, provider rerun manifests, private Formal-300 inputs, and local
  v2/v3 manifests that the documentation explicitly identifies as withheld.
- One documentation reference needs manual review:
  `README.md` says the historical audit “remains at”
  `results/answer_leakage_audit.json`, but that file is not present. The same
  historical record is described numerically in `ARTIFACTS.md`. No evidence
  file was created or altered.

### Published aggregate statistics

The aggregate JSON validation returned:

| Quantity | Value |
|---|---:|
| Benchmark-grounded bases | 731 |
| Supported base-fault pairs | 2,080 |
| Condition instances | 4,891 |
| Generating providers | 6 |
| Provider-replicated pair evaluations | 12,480 |
| Resolved native-style triplets | 298 |
| Distinct resolved bases | 238 |
| Bootstrap iterations | 10,000 |
| Bootstrap seed | 20261010 |

The six-provider summary contains OpenAI, Anthropic, DeepSeek, Google Gemini,
Kimi, and xAI. No result JSON, JSONL, CSV, model output, or analysis script
differs from `HEAD`.

### Public package manifest

- `MANIFEST.sha256` is unchanged.
- The manifest contains 454 tracked payload entries.
- 449 entries match the current working tree.
- Five entries differ, exactly the five intentionally edited metadata files:
  `README.md`, `CITATION.cff`, `ARTIFACTS.md`,
  `docs/JSS_PUBLIC_RESULTS.md`, and `pyproject.toml`.
- No other payload or manifest entry differs.
- The manifest was not rebuilt because the task explicitly prohibits modifying
  `MANIFEST.sha256` or rebuilding manifests. A maintainer must decide whether
  to authorize a separate release-manifest update after review.

### Offline tests

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -o addopts='' -q -p no:cacheprovider
```

Result:

`1 failed, 524 passed, 230 skipped in 3.27s`

The only failure is
`tests/test_jss_publication_claims.py::test_license_and_citation_metadata_are_scoped`,
which still asserts the superseded citation title. Tests were not modified
because the task explicitly prohibits test changes.

With that stale assertion deselected, the remaining public suite result was:

`524 passed, 230 skipped, 1 deselected in 3.22s`

The skips are the existing public-release exclusions for private/local or
licensed inputs. No provider API was called.

## Integrity confirmation

- `git diff --quiet -- data results third_party scripts tests`: PASS; exit 0.
- `git diff --exit-code -- MANIFEST.sha256`: PASS; exit 0.
- `git rev-parse HEAD`:
  `b0afd0792df6631096756865eb3713923d17c9a3`.
- No experiment, research result, model output, script, test, licensed
  benchmark data, or frozen JSON/JSONL/CSV artifact was modified.
- No artifact hash, research count, or published statistic was changed.

## Remaining manual review

1. Decide whether to update the stale citation-title assertion in
   `tests/test_jss_publication_claims.py`; this task prohibited test edits.
2. Decide whether to rebuild `MANIFEST.sha256` in a separately authorized
   release step after the metadata changes are approved.
3. Correct or clarify the README reference to the absent historical
   `results/answer_leakage_audit.json` without altering the underlying
   historical evidence.
4. Enter the recommended GitHub short description manually in repository
   Settings.

## Suggested review, commit, and push commands

Review:

```bash
git diff --check
git diff -- README.md CITATION.cff ARTIFACTS.md docs/JSS_PUBLIC_RESULTS.md pyproject.toml
git status --short
```

Commit after review:

```bash
git add README.md CITATION.cff ARTIFACTS.md docs/JSS_PUBLIC_RESULTS.md pyproject.toml APPLIED_INTELLIGENCE_REPO_METADATA_UPDATE.md
git commit -m "Update repository metadata for Applied Intelligence submission"
```

Push after resolving the remaining review items:

```bash
git push origin main
```
