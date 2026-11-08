# Third-party datasets, licenses, and attribution

The GovMem v3 experiments were run on three externally licensed benchmarks.
This document records, for each one, the official source, the license, the
exact version GovMem used, the required attribution, and the usage
restrictions. It also records what this release ships in order to satisfy
those obligations.

**This document is not a license for GovMem.** GovMem's own code is not
licensed by anything here; see "GovMem's own licensing" below. Nothing in this
release modifies, relicenses, or sublicenses any third-party material.

## Summary

| Dataset | Official source | License | Version GovMem used |
|---|---|---|---|
| LongMemEval (cleaned) | https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned | MIT | HF repository commit `98d7416c24c778c2fee6e6f3006e7a073259d48f`; `longmemeval_oracle.json`, SHA-256 `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c` |
| Memora | https://github.com/geniesinc/Memora | Apache License 2.0 | repository commit `a6493188efc836d6511ed5e4163fe3ba87da30ff` (2026-06-03); `data/` corpus |
| OAKS Novel | https://github.com/adobe-research/OAKS | Adobe Research License | repository commit `0bbc4b3877eae5f9335f07c5e485aabf1d9136e9` (2026-05-27); `data/oaks-novel/oaks-n.split_2k.json` |

The URLs, commits, versions, and hashes above are the values recorded in the
frozen provenance files in this package. This remediation was performed
offline. The remote commit and dataset content could not be independently
re-queried, so they are reported as repository-recorded provenance rather than
as freshly network-verified facts.

License texts shipped verbatim in this release:

| Notice file | Source | Copy SHA-256 (identical to upstream) |
|---|---|---|
| `third_party/licenses/Apache-2.0-Memora.txt` | `geniesinc/Memora` `LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `third_party/licenses/Adobe-Research-License-OAKS-Novel.txt` | `adobe-research/OAKS` `LICENSE` | `d9a8d2d8348caaab20299ef6441854cce1312dc29f67923a94f2804826751739` |
| `third_party/licenses/MIT-LongMemEval.txt` | standard MIT text (no upstream LICENSE file; card declares `license: mit`) | standard text; no upstream file to hash |

## 1. LongMemEval (cleaned)

- **Official source:** https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
  (Hugging Face repository commit `98d7416c24c778c2fee6e6f3006e7a073259d48f`,
  2025-09-19).
- **License:** MIT, declared in the dataset card front-matter
  (`license: mit`). The cleaned release "replaces the original LongMemEval
  dataset" (card README). The original LongMemEval distribution is a different
  artifact and may carry its own terms.
- **Version GovMem used:** the `longmemeval_oracle.json` split. GovMem pins the
  file by content hash, not by date:
  `official_source_sha256 = 821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`,
  recorded in `data/final_main_governance_condition_manifest_v3.freeze.json`.
  This audit re-verified that hash against the local copy of
  `longmemeval_oracle.json` and it matches byte-for-byte, so the exact version
  used is unambiguous.
- **Attribution requirement (MIT):** the copyright notice and the permission
  notice must be included in all copies or substantial portions of the
  software/data.
- **Restrictions:** none beyond the MIT notice requirement. MIT is a
  permissive license and does not restrict commercial use.
- **Open item:** the card declares `license: mit` but the repository ships no
  `LICENSE` file and no verbatim copyright-holder string. The MIT notice in
  `third_party/licenses/MIT-LongMemEval.txt` therefore carries the standard
  text with a descriptive holder line that must be confirmed or replaced with
  the authors' intended string before it is relied on beyond attribution.

**Citation** (from the GovMem manuscript bibliography; the dataset card itself
ships no BibTeX entry):

```bibtex
@inproceedings{wu2025longmemeval,
  title     = {{LongMemEval}: Benchmarking Chat Assistants on Long-Term Interactive Memory},
  author    = {Wu, Di and Wang, Hongwei and Yu, Wenhao and Zhang, Yuwei
               and Chang, Kai-Wei and Yu, Dong},
  booktitle = {International Conference on Learning Representations},
  year      = {2025}
}
```

## 2. Memora

- **Official source:** https://github.com/geniesinc/Memora
  (repository commit `a6493188efc836d6511ed5e4163fe3ba87da30ff`, 2026-06-03).
- **License:** Apache License 2.0.
  - `LICENSE` (repo root) is the unmodified Apache-2.0 text (SHA-256
    `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`).
  - `data/README.md` restates: "The dataset is released under Apache License
    2.0", so the license covers the corpus GovMem used, not only the code.
  - The repository ships **no `NOTICE` file**, so the Apache-2.0 section 4(d)
    obligation to propagate an upstream NOTICE is not triggered; there is no
    upstream NOTICE text to carry.
- **Version GovMem used:** the `data/` corpus (`weekly/`, `monthly/`,
  `quarterly/` persona directories with `conversations/` and
  `evaluation_questions_<persona>.json`). GovMem does **not** pin the upstream
  Memora corpus by per-file hash; the only related hash in the release is the
  derived-source hash `c55f625e2c5f576f6bbbaeb631f6303fd1741d7ab46a11c5fa0bff530f35cfc8`
  in `results/formal_300_validation_v1/sampling_lock.json`. Version
  identification therefore rests on the commit above and the local checkout.
- **Attribution requirements (Apache-2.0 §4):** provide a copy of the License;
  retain all copyright, patent, trademark, and attribution notices from the
  source; and, if files were modified, carry prominent notices stating that
  changes were made. GovMem builds derived manifests and does not redistribute
  any Memora file, so the practical obligations here are the license copy and
  the attribution below.
- **Restrictions:** none beyond Apache-2.0's notice/state-changes conditions.
  Apache-2.0 is permissive and does not restrict commercial use.

**Citations:**

From the Memora repository README:

```bibtex
@inproceedings{uddin2026memora,
  title     = {From Recall to Forgetting: Benchmarking Long-Term Memory
               for Personalized Agents},
  author    = {Uddin, Md Nayem and Shubham, Kumar and Blanco, Eduardo
               and Baral, Chitta and Wang, Gengyu},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for
               Computational Linguistics (ACL 2026)},
  publisher = {Association for Computational Linguistics},
  year      = {2026},
  url       = {https://arxiv.org/abs/2604.20006}
}
```

From the GovMem manuscript bibliography (same work, findings track):

```bibtex
@inproceedings{uddin2026memora,
  title     = {From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents},
  author    = {Uddin, Md Nayem and Shubham, Kumar and Blanco, Eduardo
               and Baral, Chitta and Wang, Gengyu},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages     = {26814--26841},
  year      = {2026},
  doi       = {10.18653/v1/2026.findings-acl.1337}
}
```

The two entries disagree on the venue (main conference vs findings) and the
canonical locator; cite the version that matches the venue you saw announced,
or cite both.

## 3. OAKS Novel

- **Official source:** https://github.com/adobe-research/OAKS
  (repository commit `0bbc4b3877eae5f9335f07c5e485aabf1d9136e9`, 2026-05-27).
  License file: https://github.com/adobe-research/OAKS/blob/master/LICENSE
- **License:** the **Adobe Research License** (not an OSI license). The verbatim
  text is shipped in
  `third_party/licenses/Adobe-Research-License-OAKS-Novel.txt` (SHA-256
  `d9a8d2d8348caaab20299ef6441854cce1312dc29f67923a94f2804826751739`).
  Key terms, quoted by section:
  - §1.1 (A) reproduce, use, modify, publicly display — **for noncommercial
    research purposes only**.
  - §1.1 (B) redistribute the Licensed Materials and modifications/derivative
    works — **for noncommercial research purposes only, provided that you give
    recipients a copy of this License upon redistribution**.
  - §1.3 noncommercial research = academic research and teaching only;
    excludes commercial licensing or distribution, product development, and
    any activity that results in commercial gain.
  - §2 ownership and attribution: Adobe and its licensors own all right, title,
    and interest; **you must retain all copyright notices and/or disclaimers**.
  - §3–§4 no warranty / no liability; §5 termination on material breach.
- **Version GovMem used:** `data/oaks-novel/oaks-n.split_2k.json` (the
  OAKS-Novel split). GovMem does **not** pin this file by content hash anywhere
  in the release; the only identification is the filename and this repository
  commit. This is a version-pinning gap (see open items).
- **Third-party content inside OAKS-Novel:** the upstream README states
  OAKS-Novel is "human-curated … sourced from 19 public domain novels", so the
  underlying narrative text is public domain and the Adobe license is the
  operative restriction. (The sibling `oaks-b` split is derived from BABILong
  and was **not** used by GovMem, so BABILong's terms do not apply here.)

**Citation** (from the GovMem manuscript bibliography; the upstream README
gives the title, arXiv link, and venue but no BibTeX):

```bibtex
@inproceedings{kim2026oaks,
  title     = {Can Large Language Models Keep Up? Benchmarking Online Adaptation to Continual Knowledge Streams},
  author    = {Kim, Jiyeon and Lee, Hyunji and Zhou, Dylan and Park, Sue Hyun
               and Yoon, Seunghyun and Bui, Trung and Dernoncourt, Franck
               and Cha, Sungmin and Seo, Minjoon},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {42240--42272},
  year      = {2026},
  month     = jul,
  address   = {San Diego, California, United States},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2026.acl-long.1956}
}
```

## How GovMem uses the datasets

GovMem does not redistribute any dataset file. It reads the locally obtained
licensed sources, derives governance manifests, runs the experiments, and ships
**derived aggregate statistics and labels only** (see
`docs/REPRODUCTION_TIERS.md` and `FINAL_GITHUB_RELEASE_AUDIT.md`). No benchmark
question, answer, context text, per-example identifier, or per-row fingerprint
is included. The upstream files must be obtained separately under their own
terms, as described in `docs/OBTAINING_DATASETS.md`.

## Obligations and how this release meets them

| Obligation | Applies to | How this release satisfies it |
|---|---|---|
| Include the license text | all three | `third_party/licenses/*.txt` |
| Retain copyright/attribution notices | OAKS (§2), Memora (Apache-2.0 §4) | this document + the shipped license texts |
| Give recipients a copy of the Adobe Research License | OAKS (§1.1 B) | `third_party/licenses/Adobe-Research-License-OAKS-Novel.txt`, referenced from `third_party/README.md` |
| State that changes were made | Memora (Apache-2.0 §4(b), if applicable) | not triggered: GovMem does not redistribute Memora files; derived manifests are new works described in `ARTIFACTS.md` |
| Cite the datasets | customary | citations above |
| Noncommercial-research-only use | OAKS (§1.1, §1.3) | **honoured by restricting this release to noncommercial research use; see below** |

## Redistribution basis for the staged aggregate package

The staged package contains no raw benchmark file, benchmark question, answer,
model-visible context, provider response, or per-example record. Its OAKS
material consists of aggregate labels and statistics enumerated exactly in
`OAKS_DERIVED_ARTIFACTS.md`.

The release conservatively treats those aggregates as derivative works. Adobe
Research License §1.1(B) grants redistribution of modifications and derivative
works for noncommercial research purposes when recipients receive a copy of the
license. The package satisfies those conditions by:

- limiting the OAKS-derived material to noncommercial academic research;
- shipping the verbatim Adobe license; and
- retaining the upstream ownership and attribution statements.

No OAKS-derived data file was excluded. The applicable license permits the
aggregate redistribution under those conditions, and retaining the files
preserves the manuscript's published results.

## Remaining provenance issues

1. **OAKS-Novel's noncommercial restriction applies to its derivative
   aggregates.** The release treats the listed aggregate files as derivative
   works and relies on §1.1(B), which permits noncommercial research
   redistribution with a license copy. A permissively reusable release would
   require dropping these OAKS-derived aggregates or obtaining separate
   clearance.
2. **No OAKS-Novel source hash is pinned** in the release. The exact upstream
   revision used is identified only by filename and repository commit. Pin the
   hash before publishing, or state the revision in the release.
3. **LongMemEval MIT is declared only in the dataset card**; there is no
   `LICENSE` file and no verbatim copyright-holder string. Confirm the holder
   string before relying on the MIT notice beyond attribution.
4. **Memora's license covers the dataset** (`data/README.md`), and no upstream
   `NOTICE` file exists. No further action is required, but if Memora publishes
   a NOTICE later it must be propagated.
5. **GovMem ownership confirmation is recorded.** `LICENSE_APPROVAL.md` records
   the public-release and code-MIT instruction from all four coauthors and the
   subsequent confirmation that GovMem is independent personal research with
   no employer, university, funder, or contractor restriction on the original
   materials. Original documentation, configuration, and aggregate results are
   covered by `MATERIALS_REDISTRIBUTION_NOTICE.md`, not MIT.

## GovMem's own licensing

`LICENSE` is an MIT grant for original GovMem implementation and test code
only. Its scope is defined in `LICENSE_SCOPE.md`. It does not apply to
third-party data, third-party license texts, benchmark-derived aggregate
results, or prose documentation. Original documentation, configuration, and
aggregate distribution is addressed separately in
`MATERIALS_REDISTRIBUTION_NOTICE.md`. The third-party license texts do not
grant rights to GovMem's own material.
