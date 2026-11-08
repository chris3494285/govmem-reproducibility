# Third-party notices

This directory carries the license notices and attribution for the three
externally licensed benchmarks that the GovMem v3 experiments were run on.
They are **upstream third-party licenses**, reproduced so that this release
satisfies its attribution and license-copy obligations. They are not, and do
not imply, a license for GovMem's own code or results.

The root `LICENSE` is an MIT grant scoped **only** to original GovMem
implementation and test code. It does not cover these third-party notices,
benchmark files, or benchmark-derived aggregate material. See
`LICENSE_SCOPE.md`, `MATERIALS_REDISTRIBUTION_NOTICE.md`,
`THIRD_PARTY_DATASETS.md`, and
`OAKS_DERIVED_ARTIFACTS.md` for the complete scope and provenance record.

## Files

| File | Upstream source | License |
|---|---|---|
| `licenses/Apache-2.0-Memora.txt` | `geniesinc/Memora` (repo `LICENSE`) | Apache License 2.0 |
| `licenses/Adobe-Research-License-OAKS-Novel.txt` | `adobe-research/OAKS` (repo `LICENSE`) | Adobe Research License |
| `licenses/MIT-LongMemEval.txt` | `xiaowu0162/longmemeval-cleaned` (dataset card, `license: mit`) | MIT License |

Each file's SHA-256, and the upstream commit/artifact it was taken from, is
recorded in `THIRD_PARTY_DATASETS.md`.

Note on `licenses/MIT-LongMemEval.txt`: the LongMemEval-cleaned dataset card
declares `license: mit` but the upstream repository ships no `LICENSE` file and
no verbatim copyright-holder string. The file below is therefore the standard,
unmodified MIT text; its copyright-holder line is a description rather than a
quotation and should be confirmed with the dataset authors. The other two files
are byte-identical copies of the upstream license files.
