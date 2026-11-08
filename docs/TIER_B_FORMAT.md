# Tier-B pseudonymous score format

> **Not shipped in the current (aggregate-only) release.** The six Tier-B files
> under `results/final_main_tier_b/` are a developer-tree artifact. Tier-B was
> an improvement over `results/final_main_sanitized/`, but it still retained a
> row/condition structure derived from the licensed benchmarks, so the
> aggregate-only release withholds it too and ships only aggregate tables and
> summaries. This document is retained to record the Tier-B field policy and
> why it was superseded. See `docs/REPRODUCTION_TIERS.md` for what is shipped.

Historically, the canonical public v3 score artifacts were the six Tier-B files
under `results/final_main_tier_b/`. They replaced the earlier
`results/final_main_sanitized/` files, which retained source-derived benchmark
identifiers and per-row content fingerprints whose redistribution rights are
not established.

## What changed relative to the earlier sanitized files

| Field | Tier-B treatment |
|---|---|
| `base_id` | Replaced by a release-local label `b0001…` |
| `condition_id` | Replaced by a release-local label `c00001…` |
| `pair_id` | Replaced by a release-local label `p00001…` (or `null`) |
| `model_visible_context_sha256` | **Removed** |
| `prompt_sha256` | **Removed** |
| `provider`, `model`, `model_parameters` | Retained |
| `condition_type`, `fault_operator`, `analysis_event_type`, `primary_event_type` | Retained (GovMem-defined taxonomy) |
| `source_dataset` | Retained (benchmark name only — not an identifier) |
| `status`, `parse_success`, `scoring_status`, `scoring_version`, `scoring_method` | Retained |
| `correctness`, `primary_correctness`, `primary_score` | Retained |
| `manifest_sha256` (whole-file provenance) | Retained — already published in `data/*.freeze.json` |
| `attempt_count`, `retry_count`, `recovered_by_retry`, token/latency accounting | Retained |

## The pseudonymisation rule

Distinct original identifiers are sorted lexicographically and numbered:

- `base_id` → `b0001…` over the 731 distinct bases;
- `condition_id` → `c00001…` over the 4,891 distinct conditions;
- `pair_id` → `p00001…` over the 2,080 distinct base-fault pairs.

The same rule and the same mapping are applied to every other released artifact
that mentions a condition, base, or pair id, so cross-file references stay
consistent inside the release (`scripts/build_public_release_candidate.py`,
`identifier_pseudonymization`).

The **original→pseudonym mapping is private** and is never copied into the
release. It is held outside the release directory and is passed to the builder
via `--id-mapping`.

## Limitations (reduction in reproducibility)

**Still possible from the public artifact alone**

- Recomputing every six-provider aggregate statistic: per-provider corruption
  and recovery effects with bootstrap CIs, the control-plane change/restoration
  diagnostic, matched-fault support and overlap counts, and the
  clean-repeat/formal-300 summaries. All required non-identifier fields are
  retained, and the frozen aggregate outputs are shipped alongside.
- Verifying the row count, provider/source/condition/fault composition, and
  scoring-status distribution.

**No longer possible from the public artifact alone**

- Joining a released row to a specific benchmark example, session, or document
  (the source-derived ids are gone).
- Verifying per-row `context`/`prompt` fingerprints against a rebuilt manifest
  (those fingerprints are removed).
- Reconstructing the original→pseudonym mapping.

**Possible for a party that holds the licensed datasets and the private mapping**

- Rebuilding the mapping exactly by applying the published ordering rule to a
  locally rebuilt manifest, then recovering the join. `scripts/verify_tier_b_release.py`
  performs exactly this and confirms the aggregates are unchanged.

## Verification

```bash
# rebuild the six-provider aggregates from Tier-B inputs and compare to frozen
python scripts/verify_tier_b_release.py \
  --tier-b-dir results/final_main_tier_b \
  --mapping /local/private/id_mapping.json \
  --manifest /local/licensed/final_main_governance_condition_manifest_v3.jsonl
```

Result of the frozen check: all six provider `primary` summaries, the
control-plane `overall` diagnostic, and all matched-fault statistics
(`matched_instance_count`, `overlap_counts`, `comparisons`, `support_counts`)
recompute **identically** to the published frozen values.

## Residual uncertainty

Tier-B removes source-derived identifiers and content fingerprints from the
score artifacts, but the retained `source_dataset` labels, the GovMem taxonomy
labels, and the row/condition structure are still *derived from* the licensed
benchmarks. Redistribution rights for those derived structural facts are not
confirmed by this repository; confirm them with each benchmark's terms before
publishing.
