# GovMem license scope

`LICENSE` is the GovMem MIT license for original implementation and test code
only.

## Covered

- Original Python source under `govmem/`, `eval/`, `baselines/`, and `scripts/`
  that contains only GovMem-authored code.
- Original Python test code under `tests/`.

The code may process benchmark data, but the MIT grant does not extend to any
benchmark material or derived material governed by separate terms.

## Not covered

- `third_party/**`, including all upstream license texts and notices.
- LongMemEval, Memora, or OAKS Novel data, identifiers, text, or other
  third-party material.
- Benchmark-derived aggregate data and result files, including the OAKS-derived
  artifacts listed in THIRD_PARTY_DATASETS.md.
- Original prose documentation and reports, unless a separate written decision
  explicitly places them under `LICENSE`.
- Original configuration specifications in `configs/`.
- `CITATION.cff`, dataset citations, and third-party bibliography entries.

No file under `third_party/` is relicensed by the GovMem MIT file.

## Approval status

The four coauthors confirmed on 2026-11-08 that they agree to the public release
and that the intended license for original GovMem code is MIT. The recorded
instruction is preserved in `LICENSE_APPROVAL.md`.

That approval does not establish:

1. that every contributor was an employee or contractor acting with the same
   institutional rights;
2. that an employer, university, laboratory, or funder does not hold rights in
   any covered contribution; or
3. that every code file was authored solely by the four named coauthors.

The responsible humans subsequently confirmed that GovMem is independent
personal research and that no employer, university, funder, or contractor owns
or restricts the original research materials. That confirmation is recorded in
`LICENSE_APPROVAL.md`.
