# Source × Event × Fault Distribution of the Frozen GovMem v3 Cohort

Offline audit of the frozen final v3 experimental artifacts. No model or paid API was called, no response was regenerated, and no frozen manifest, metric, or paper number was modified. All figures below are recomputed from the frozen manifests; no cell is estimated.

## 1. Authoritative inputs

| Role | File | SHA-256 |
| --- | --- | --- |
| condition manifest | `final_main_governance_condition_manifest_v3.jsonl` | `4b84068a341d912f43d07c9d92059f75224ea2636abf69e40ba6e71a0a1144cb` |
| base manifest | `final_main_governance_base_manifest_v2.jsonl` | `a2601a7ecb46034faf5dcc7ad00b5a851e19ae01561876fa681fd18af168ca79` |
| freeze metadata | `final_main_governance_condition_manifest_v3.freeze.json` | `12b9a27585e9e40af8666cfef2573e8492aa7bfa504a10a85b81e798be12b941` |

The condition manifest is the authoritative frozen final v3 cohort (`main-governance-condition-v3` rows supersede the v2 rows for 53 repaired bases; the freeze file records the v2 digest that v3 supersedes). The base manifest is the parent cohort manifest and supplies the per-base `supported_fault_operators` / `unsupported_fault_operators` applicability declarations.

## 2. Counting units

- **Base example** — one frozen benchmark example (`base_id`).
- **Matched fault pair** — one (base example, fault operator) cell with exactly one `corrupted` and one `enforced` condition sharing a `pair_id`.
- **Condition** — one manifest record (`condition_id`): one `clean` record per base plus one `corrupted` and one `enforced` record per matched fault pair.

Columns in the cross-tabulation:

- `distinct_base_count` — number of distinct bases in the cell.
- `fault_pair_count` — number of matched fault pairs in the cell.
- `corrupted_condition_count` — corrupted condition rows in the cell.
- `enforced_condition_count` — enforced condition rows in the cell.

Because each matched fault pair carries exactly one corrupted and one enforced sibling, `corrupted_condition_count == enforced_condition_count == fault_pair_count` in every observed cell; the equality is re-checked rather than assumed. `distinct_base_count` is a de-duplicated base count, so it is not additive across faults; `fault_pair_count` is additive.

## 3. Global reconciliation

| Quantity | Frozen cohort |
| --- | --- |
| Distinct base examples | 731 |
| Matched fault pairs | 2080 |
| Corrupted conditions | 2080 |
| Enforced conditions | 2080 |
| Clean conditions | 731 |
| Total conditions | 4891 |

Reconstruction: 731 clean + 2080 corrupted + 2080 enforced = 4891 total conditions; 2080 + 2080 = 2 × 2080 matched-pair conditions.

## 4. Source × Event × Fault table

| Source | Event | Bases | Missing | Stale | Flipped | Conflicting |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LongMemEval | Deletion | 0 | n/a | n/a | n/a | n/a |
| LongMemEval | Temporal | 113 | 113 | 113 | -- | -- |
| LongMemEval | Update | 78 | 78 | 78 | -- | 78 |
| Memora | Deletion | 380 | 380 | -- | 380 | 380 |
| Memora | Temporal | 0 | n/a | n/a | n/a | n/a |
| Memora | Update | 0 | n/a | n/a | n/a | n/a |
| OAKS Novel | Deletion | 0 | n/a | n/a | n/a | n/a |
| OAKS Novel | Temporal | 56 | 56 | 56 | -- | 56 |
| OAKS Novel | Update | 104 | 104 | 104 | -- | 104 |
| **Total** | | **731** | **731** | **351** | **380** | **618** |

Entries are matched fault pairs. `--` = fault operator declared unsupported for that source × event; `n/a` = source × event combination absent from the cohort.

## 5. Empty cells: structurally inapplicable vs observed zero

The full grid is 3 sources × 3 events × 4 fault operators = 36 cells. The frozen cohort populates 14 of them. The remaining cells are empty for two structurally different reasons, and they are **not** interchangeable:

1. **Absent source × event combinations (4):** LongMemEval × Deletion, Memora × Temporal, Memora × Update, OAKS Novel × Deletion. These 4 combinations have no bases in the frozen cohort, so all 16 of their cells are inapplicable. Event is not crossed with every source: **deletion exists only for Memora**.
2. **Declared-unsupported fault operators (6):** LongMemEval × Temporal × Flipped, LongMemEval × Temporal × Conflicting, LongMemEval × Update × Flipped, Memora × Deletion × Stale, OAKS Novel × Temporal × Flipped, OAKS Novel × Update × Flipped. Here the source × event stratum exists, but the fault operator is listed in the frozen base manifest's `unsupported_fault_operators` for every base in the stratum, so the cell is structurally inapplicable by construction.

**Observed zero-count cells (supported in principle but empty in practice): 0.** The frozen cohort contains no cell that is supported by declaration yet unpopulated. This distinction matters because only the last category would be a genuine data gap; the other two are properties of the frozen design and are reported as such.

## 6. Fault multiplicity per base and the proposed identity

Bases by number of applicable fault operators: 2 faults → 113 bases, 3 faults → 618 bases.

The proposed identity **2080 = 618 * 3 + 113 * 2** is **supported** by the frozen data: 618 bases support 3 fault operators (1854 pairs) and 113 bases support 2 (226 pairs); 1854 + 226 = 2080 = observed matched fault pairs, and 618 + 113 = 731 = observed bases.

The 113 two-fault bases are exactly the LongMemEval × Temporal stratum, which supports only {missing, stale}; every other stratum supports three fault operators.

The identity is therefore a statement about **fault applicability per base**, not about a fixed number of faults per event. An alternative model in which every base receives two faults per event would predict deletion 760, temporal 338, update 364 pair rows; the frozen cohort instead has deletion 1140, temporal 394, update 546 matched fault pairs, so that model is not the frozen construction.

## 7. Source and Event are confounded

Base counts by source × event (rows sum to the source totals, columns to the event totals):

| Source | Deletion | Temporal | Update | Total |
| --- | ---: | ---: | ---: | ---: |
| LongMemEval | 0 | 113 | 78 | 191 |
| Memora | 380 | 0 | 0 | 380 |
| OAKS Novel | 0 | 56 | 104 | 160 |
| **Total** | **380** | **169** | **182** | **731** |

Event is **not** crossed with source: Deletion occurs only for Memora; Temporal occurs only for LongMemEval, OAKS Novel; Update occurs only for LongMemEval, OAKS Novel. In particular deletion is the exclusive event of Memora and Memora has no temporal or update bases, so a deletion-vs-other contrast is numerically identical to a Memora-vs-other contrast. Association statistic on the source × event table: χ² = 773.4 (dof = 4), Cramér's V = 0.727 — a strong, design-imposed dependence. Event-level effects must therefore be reported as descriptive of this frozen cohort and must not be interpreted as independently identified causal differences between event types. Producing this table documents the confounding; it does not remove it.

## 8. Integrity checks performed

| Check | Result |
| --- | --- |
| Unique `condition_id` | 4891 |
| Unique `base_id` | 731 |
| `condition_type` counts | {'clean': 731, 'corrupted': 2080, 'enforced': 2080} |
| Bases with exactly one clean condition | pass |
| Fault pairs with complete corrupted+enforced siblings | 2080 |
| Orphaned sibling conditions / duplicate fault pairs | 0 / 0 |
| Conditions with no matching base | 0 |
| Bases with inconsistent source/event labels | 731 consistent |
| Declared fault-applicability violations | 0 |
| Base-manifest SHA mismatches inside conditions | 0 |

Any breach of these checks aborts the analysis with an error rather than emitting partial output.

## 9. How to read this table

- Every number is a frozen count recomputed from the manifests; nothing is estimated or imputed.
- `--` and `n/a` are structural, not measured: they encode the frozen design's applicability declaration and the source × event coverage, not a null result.
- The three sources are not balanced: Memora supplies all deletion bases and roughly half of all matched fault pairs, while LongMemEval is the only source with a two-fault stratum.
- Because source and event are confounded, per-event and per-source effects in the paper describe the same partition of the cohort; use them descriptively.
