# Obtaining the licensed datasets and rebuilding experiment inputs

The GovMem v3 experiments use three externally licensed benchmarks. **No raw
benchmark text, task question, reference answer, or model-visible context text
is redistributed in this release**, and no derived identifier or fingerprint is
released without an explicit licence review. To reproduce the aggregate
results you must obtain each dataset from its official source, accept its
terms, and rebuild the derived manifests locally.

This document is the reader-facing acquisition guide. It does not grant any
right to redistribute the datasets or derived identifiers.

## 1. LongMemEval (191 bases in the final cohort)

- **Official repository:**
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- **Repository-recorded immutable revision:**
  `98d7416c24c778c2fee6e6f3006e7a073259d48f`.
- **Official source used for v3:** the cleaned LongMemEval oracle file
  `longmemeval-cleaned/longmemeval_oracle.json`.
- **License:** MIT, as declared in the cleaned dataset card. The repository
  does not publish a verbatim copyright-holder string; confirm the canonical
  notice with the dataset authors before treating the attribution as final.
- **Frozen identity:** `data/final_main_governance_condition_manifest_v3.freeze.json`
  records `official_source_locator` and
  `official_source_sha256 = 821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`.
  Verify your local copy with `shasum -a 256`; a different file is not the
  frozen source and results will not match.
- **Where to get it:** the LongMemEval project repository/leaderboard release
  (see the LongMemEval paper and its official distribution channel). Obtain it
  yourself under its own terms; do not copy the file from this repository —
  it is not included.
- **Expected shape:** a JSON array whose items carry, at minimum, `id`,
  `question`, `answer`, `memories`, `user_id`, `current_time`, plus the
  `dataset_mode`/`allowed_scopes` fields used by the adapter
  `eval/official_longmemeval.py`.
- **Rebuild the repaired v3 evidence locally:**

  ```bash
  export LONGMEMEVAL_SOURCE=/path/to/longmemeval-cleaned
  python scripts/repair_answer_leakage.py \
    --source "$LONGMEMEVAL_SOURCE/longmemeval_oracle.json"
  python scripts/audit_answer_leakage.py   # default target is the repaired v3 manifest
  ```

  The v3 manifest is written under `data/` and is **not** part of the public
  release; only the freeze metadata and hashes are.

## 2. Memora (380 bases in the final cohort)

- **Official repository:** https://github.com/geniesinc/Memora
- **Repository-recorded immutable revision:**
  `a6493188efc836d6511ed5e4163fe3ba87da30ff`.
- **Official source used:** the Memora data directory (`Memora/data`).
- **License:** Apache License 2.0. The full license copy is included under
  `third_party/licenses/Apache-2.0-Memora.txt`.
- **Where to get it:** the Memora project’s official release. Obtain it under
  its own terms.
- **Expected shape:** a persona directory tree; each persona/period directory
  contains exactly one `evaluation_questions_<persona>.json` file, plus the
  supporting conversation/session files. See `eval/memora_adapter.py` for the
  exact schema the adapter accepts (`questions`, `memory_evidence`,
  `forgetting_evidence`, `evaluation.evaluation_questions`, …).
- **Rebuild locally:**

  ```bash
  export MEMORA_SOURCE=/path/to/Memora/data
  ```

  The manifest and B4 builders take `--memora-source "$MEMORA_SOURCE"`; they
  read the licensed files directly and write only text-free derived artifacts
  to `results/`.

## 3. OAKS Novel (160 bases in the final cohort)

- **Official repository:** https://github.com/adobe-research/OAKS
- **Repository-recorded immutable revision:**
  `0bbc4b3877eae5f9335f07c5e485aabf1d9136e9`.
- **Official source used:** the OAKS Novel release file
  `oaks-novel/oaks-n.split_2k.json`.
- **License:** Adobe Research License. It permits use and redistribution of
  derivative works only for noncommercial academic research/teaching and
  requires recipients to receive a copy of the license. The full license is
  included under `third_party/licenses/Adobe-Research-License-OAKS-Novel.txt`.
- **Where to get it:** the OAKS project’s official release. Obtain it under its
  own terms.
- **Expected shape:** a list of document objects each with a native `meta`
  object (including `bid`), ordered context `chunks`, and a `questions`
  list/map. See `scripts/audit_oaks_novel_for_main_dataset.py`, which validates
  the structure and derives deterministic transition evidence.
- **Rebuild locally:**

  ```bash
  export OAKS_NOVEL_SOURCE=/path/to/oaks-novel/oaks-n.split_2k.json
  python scripts/audit_oaks_novel_for_main_dataset.py
  ```

  This writes a candidate audit (not included here) that the frozen base
  manifest builder consumes.

The staged package's complete inventory of OAKS-derived aggregate artifacts is
listed in `OAKS_DERIVED_ARTIFACTS.md`. Raw OAKS content is not included.

## 4. Rebuild the frozen base and condition manifests

Once all three sources are available locally:

```bash
python scripts/build_final_main_governance_manifest_v2.py     # base cohort
python scripts/build_final_main_governance_condition_manifest_v2.py  # clean/corrupted/enforced
```

Confirm the reproduced manifests against the frozen hashes before analysing:

- base manifest `a2601a7ecb46034faf5dcc7ad00b5a851e19ae01561876fa681fd18af168ca79`
- condition manifest v2 `58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8`
- condition manifest v3 `4b84068a341d912f43d07c9d92059f75224ea2636abf69e40ba6e71a0a1144cb`

The freeze files `data/*.freeze.json` in this release are the machine-readable
record of those hashes, row counts, and source identities, and are safe to
publish.

## 5. What is safe to share downstream

This release is **aggregate-only**. The following are **not** part of it and
must not be published without a separate licence review: task questions,
reference/oracle answers, model-visible context text, provider response bodies,
parsed answers, request identifiers, endpoint/account metadata, any
benchmark-derived text-bearing manifest or rerun payload, **and every
condition-level (row-level) score file** — including the earlier Tier-B
pseudonymous files, which still carried a row/condition structure derived from
the licensed benchmarks.

What the release does contain is a set of aggregate tables and summaries
(overall, and by provider, model, source dataset name, event type, fault
operator, and condition type), the analysis code and frozen non-secret specs
that produced them, and the freeze metadata and hashes that pin the local
manifests. That is enough to verify the published aggregates (Tier 1) and to
recompute them after independently rebuilding the licensed inputs (Tier 2); it
is not enough for any per-example analysis (Tier 3). See
`docs/REPRODUCTION_TIERS.md`.
