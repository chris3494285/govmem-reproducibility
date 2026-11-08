# GovMem B4 — Independent Scoring Evidence Gap Audit

Audit date: 2026-10-15. Repository: the original private experimental tree
(`govmem-demo` main worktree plus its linked `govmem-public-release` worktree,
which share one `.git`). Scope: does the existing Formal-300 benchmark-native-style
validation adequately address Youting's B4 scorer-validity concern?

Provenance policy for this document: read-only with respect to every frozen
manifest, scoring decision, model response, and paper number. **Zero** LLM, API,
network, scoring, judging, or answer-generation calls were made. All numbers are
recomputed from frozen artifacts; nothing is estimated, imputed, or resampled.
Only aggregate counts and already-public identifier hashes are reproduced.

Reproduction: `python scripts/audit_b4_evidence_gap.py` (offline; reads the six
frozen score files, the frozen v3 manifests, and the frozen Formal-300 selection).

---

## 0. Verdict summary

| Claim | Adequately supported by existing evidence? | Basis |
|---|---|---|
| **A.** Limited claim that governance corruption degrades downstream task quality | **Partially — CONDITIONAL** | 298/300 matched triplets re-scored under a native-style evaluator; corruption +35.5 pp [29.8, 41.1], recovery +32.7 pp [26.9, 38.4]. Small, non-probability sample; no OAKS in the semantic leg; magnitudes differ from the frozen scorer. |
| **B.** Population-level semantic effect estimate for all six providers | **NO** | 300 of 12,480 provider–pair units (2.40%) score under the native-style evaluator; 0 of 2,880 OAKS provider–pair units; selection is deterministic and provider-balanced by construction, not a probability sample. Per-provider CIs are not even computed. |
| **C.** Validation of the full frozen-scorer effect magnitudes | **NO** | Formal-300 never evaluates the frozen deterministic scorer; it replaces it. The frozen-scorer validity evidence that does exist (604-case B4 cohort) shows 45.4% agreement with an independent semantic reference (κ = 0.114) and only 96 of 604 cases are same-unit. |

A does not imply B, and A does not imply C. Recommendation: **CONDITIONAL GO**
(see §7).

---

## 1. Evidence inventory

| Evidence block | Artifact root | Protocol / evaluator | Status | Calls made to produce it |
|---|---|---|---|---|
| Formal-300 behavioral validation | `results/formal_300_validation_v1/` (+ `results/cheap_judge_pilot_v1/`) | `formal-300-validation-v1`, `final-native-evaluation-v2` | **Complete** | 907 judge requests + 1 human adjudication |
| B4 scorer calibration (cheap judges) | `results/b4_scoring_validation/private/cheap_judges`, `.../calibration` | `b4-semantic-judge-calibration` | Failed selection gate → exploratory only | DeepSeek 639 + Kimi 664 results |
| B4 frontier 604-case semantic reference | `results/b4_scoring_validation/private/full_frontier_judges`, `.../frontier_escalation`, `.../frontier_adjudication` | dual frontier primaries + third judge + human | **Complete (604 resolved)** | 1,208 primary slots + 24 escalation |
| B4 deterministic-scorer validation | `results/b4_scoring_validation/final_scorer_validation/` | `b4-scorer-validation-v1` | Complete, **but Memora not unit-aligned** | 0 (analysis only) |
| B4 unit-alignment audit | `results/b4_scoring_validation/unit_alignment_audit/` | `b4-scorer-unit-alignment-audit-v1` | Complete | 0 |
| OAKS native parity | `results/b4_scoring_validation/oaks_validation.json`, `oaks_input_summary.json` | `oaks-native-parity-v1` | Complete | 0 (native option comparison) |
| Full native re-evaluation plan | `results/full_native_reevaluation_v1/` | `full-native-reevaluation-v1` | **Planned, never executed** | 0 performed / 22,727 planned |
| Paper scorer-impact audit | `results/b4_scoring_validation/paper_impact_audit/` | `b4-paper-scorer-impact-v1` | Complete | 0 |

Frozen score inputs (SHA-256 prefixes verified against `sampling_lock.json`):
`final_main_openai_gpt56_sol_final_v2` `cd69b5e4…`, `final_main_anthropic_sonnet5_final`
`d386ad9a…`, `final_main_deepseek_v4_pro` `e851230a…`, `final_main_gemini_3_flash_final`
`1f1a7bbb…`, `final_main_kimi_k2_6_nothinking_final` `d2e431c8…`, `final_main_grok_43_final`
`73c72023…`. Each holds 4,891 rows.

---

## 2. Formal-300 protocol recovery

### 2.1 What a matched triplet is

The sampling unit of the pilot/formal design is **one unique (generating provider,
base example) pair carrying one matched fault pair**. Practically, one *block* is
the triple of answers to the same base question:

- `clean` — the unmodified memory condition,
- `corrupted` — a governance fault applied to that base,
- `enforced` — the same fault after control-plane enforcement.

One block = 3 observations, bound by a `pilot_block_id` and a `pair_id` of the form
`base_id::fault_operator`. 300 blocks = 900 observations. Text never appears in the
formally shipped artifacts; identities are hashes.

### 2.2 How the 300 cases were selected

1. Candidates are built offline from the six frozen `full_results_scored.jsonl`
   files restricted to **LongMemEval and Memora only** (`build_candidates` in
   `scripts/prepare_cheap_judge_pilot.py`): a candidate exists only when a
   (provider, base) group has exactly one valid `clean` row plus a complete
   `corrupted`+`enforced` pair, and all three rows pass `is_valid_answer_row`
   (`status == "success"`, parsed, non-empty).
2. For each of the 12 (provider × source) cells, `select_blocks` performs a
   **deterministic greedy stratified draw** of **25 unique bases** per cell. At each
   step it takes the candidate that minimizes `(fault_count, event_count,
   sha256(seed, provider, source, base_id, pair_id))`. The seed is a fixed constant
   `20261010`. The `used_bases` set forces one base per cell to be used at most once.
3. The 300 selected blocks were then **frozen** and fingerprinted
   `<FORMAL_BLOCK_FINGERPRINT>`.
4. Formal-300 (`prepare_formal_300_validation.validate_lock`) re-reads exactly those
   blocks, asserts the fingerprint, block count (300), observation count (900),
   triplet cardinality (3 each), source split (150/150) and provider split (50 each),
   and **resamples nothing**.

### 2.3 Random, stratified, or convenience-based?

**Deterministic, seed-keyed, greedy stratified, with a unique-base constraint — not
a probability sample and not ad-hoc convenience selection.** Two properties matter:

* It is *reproducible*: the same frozen seed and inputs regenerate the identical 300.
* It is *balanced by construction*: 25 per (provider × source) and roughly equal
  fault/event use, so observed stratum ratios are design artifacts, not cohort
  proportions, and therefore cannot be read as population estimates.

### 2.4 Providers, datasets, events, faults

| Dimension | Formal-300 |
|---|---|
| Providers | openai (`gpt-5.6-sol`), anthropic (`claude-sonnet-5`), deepseek (`deepseek-v4-pro`), gemini (`gemini-3-flash`), kimi (`kimi-k2.6-nothinking`), xai (`grok-4.3`) — 50 blocks each |
| Datasets | LongMemEval 150, Memora 150, **OAKS Novel 0** |
| Events | deletion 150, temporal 73, update 77 |
| Faults | missing 101, stale 51, flipped 50, conflicting 98 |
| Distinct bases | 239 (188 used once, 41 twice, 10 three times) |

Because source and event are confounded in the cohort (deletion exists only for
Memora), the event table is really Memora×{deletion} plus LongMemEval×{temporal,
update}: LongMemEval contributes temporal 73 / update 77, Memora contributes
deletion 150. Likewise LME carries the two-fault stratum {missing, stale} plus
conflicting, while Memora carries {missing, flipped, conflicting}.

### 2.5 Exclusions and how many

Recomputed offline with the frozen sampler code path (see §3.3 for the table):

| Exclusion | Units |
|---|---:|
| OAKS Novel removed by protocol ("frozen native deterministic parity already complete") | 2,880 provider–pair units / 960 provider–base units |
| Ineligible LME+Memora provider–pair units (incomplete/failed answer triplet) | 9 of 9,600 |
| Eligible LME+Memora provider–pair units not selected | 9,291 of 9,591 (96.87%) |
| Eligible LME+Memora provider–base units not selected | 3,125 of 3,425 (91.24%) |
| Blocks dropped at analysis time | 2 of 300 (unresolved: 2 LongMemEval primaries returned `adjudication_required`) |

Per-provider answer validity is why the eligible pool is not exactly 3,771 rows:
anthropic 3,766, openai 3,769, deepseek 3,769, gemini 3,771, kimi 3,771, xai 3,771
valid LongMemEval+Memora conditions.

### 2.6 Do the same base examples appear in multiple cases?

Yes. Bases are unique **within** a (provider × source) cell but are **reused across
providers**: 239 distinct bases cover 300 blocks. This is intentional —
provider effects are measured on partially shared content — but it means the 300
blocks are not 300 independent content draws, and any provider comparison is
partially a within-base comparison.

---

## 3. Coverage: Formal-300 vs the full frozen v3 cohort

Two denominators are used and never mixed.

* **provider–pair unit** = (provider, base, fault operator). Full cohort:
  2,080 pairs × 6 providers = **12,480**.
* **provider–base unit** = (provider, base). Full cohort: 731 bases × 6 = **4,386**.

Formal-300 contributes 300 provider–pair units and 300 provider–base units
(exactly one fault pair per block).

### 3.1 By provider

| Provider | Formal-300 | v3 pairs (1 provider) | coverage | v3 bases (1 provider) | coverage |
|---|---:|---:|---:|---:|---:|
| openai | 50 | 2,080 | 2.40% | 731 | 6.84% |
| anthropic | 50 | 2,080 | 2.40% | 731 | 6.84% |
| deepseek | 50 | 2,080 | 2.40% | 731 | 6.84% |
| gemini | 50 | 2,080 | 2.40% | 731 | 6.84% |
| kimi | 50 | 2,080 | 2.40% | 731 | 6.84% |
| xai | 50 | 2,080 | 2.40% | 731 | 6.84% |
| **All six** | **300** | **12,480** | **2.40%** | **4,386** | **6.84%** |

### 3.2 By dataset (provider-replicated)

| Dataset | Formal-300 | v3 provider–pair units | coverage | v3 provider–base units | coverage |
|---|---:|---:|---:|---:|---:|
| LongMemEval | 150 | 2,760 | 5.43% | 1,146 | 13.09% |
| Memora | 150 | 6,840 | 2.19% | 2,280 | 6.58% |
| OAKS Novel | 0 | 2,880 | 0.00% | 960 | 0.00% |
| **Total** | **300** | **12,480** | **2.40%** | **4,386** | **6.84%** |

### 3.3 By event (provider-replicated)

| Event | Formal-300 | v3 provider–pair units | coverage | v3 provider–base units | coverage |
|---|---:|---:|---:|---:|---:|
| deletion | 150 | 6,840 | 2.19% | 2,280 | 6.58% |
| temporal | 73 | 2,364 | 3.09% | 1,014 | 7.20% |
| update | 77 | 3,276 | 2.35% | 1,092 | 7.05% |

### 3.4 By fault operator (provider-replicated)

| Fault | Formal-300 | v3 provider–pair units | coverage |
|---|---:|---:|---:|
| missing | 101 | 4,386 | 2.30% |
| stale | 51 | 2,106 | 2.42% |
| flipped | 50 | 2,280 | 2.19% |
| conflicting | 98 | 3,708 | 2.64% |

### 3.5 provider × dataset

| Provider | Dataset | Formal-300 | v3 provider–pair units | coverage |
|---|---|---:|---:|---:|
| all six (each) | LongMemEval | 25 | 2,760 | 0.91% |
| all six (each) | Memora | 25 | 6,840 | 0.37% |
| any provider | OAKS Novel | 0 | 2,880 | 0.00% |

### 3.6 provider × fault

| Provider | missing | stale | flipped | conflicting | Total |
|---|---:|---:|---:|---:|---:|
| openai | 16 | 9 | 8 | 17 | 50 |
| anthropic | 17 | 8 | 8 | 17 | 50 |
| deepseek | 18 | 8 | 8 | 16 | 50 |
| gemini | 16 | 9 | 9 | 16 | 50 |
| kimi | 17 | 9 | 8 | 16 | 50 |
| xai | 17 | 8 | 9 | 16 | 50 |
| v3 provider–pair units | 4,386 | 2,106 | 2,280 | 3,708 | 12,480 |
| coverage (each row) | 0.36–0.41% | 0.38–0.43% | 0.35–0.39% | 0.43–0.46% | 0.40% |

### 3.7 Eligible-vs-selected accounting (offline recomputation)

| Provider | Source | eligible bases | eligible pairs | selected | sel/eligible base | sel/eligible pair |
|---|---|---:|---:|---:|---:|---:|
| openai | LongMemEval | 191 | 460 | 25 | 13.09% | 5.43% |
| openai | Memora | 380 | 1,138 | 25 | 6.58% | 2.20% |
| anthropic | LongMemEval | 190 | 456 | 25 | 13.16% | 5.48% |
| anthropic | Memora | 380 | 1,139 | 25 | 6.58% | 2.19% |
| deepseek | LongMemEval | 191 | 460 | 25 | 13.09% | 5.43% |
| deepseek | Memora | 380 | 1,138 | 25 | 6.58% | 2.20% |
| gemini | LongMemEval | 191 | 460 | 25 | 13.09% | 5.43% |
| gemini | Memora | 380 | 1,140 | 25 | 6.58% | 2.19% |
| kimi | LongMemEval | 191 | 460 | 25 | 13.09% | 5.43% |
| kimi | Memora | 380 | 1,140 | 25 | 6.58% | 2.19% |
| xai | LongMemEval | 191 | 460 | 25 | 13.09% | 5.43% |
| xai | Memora | 380 | 1,140 | 25 | 6.58% | 2.19% |
| **Total** | | **3,425** | **9,591** | **300** | **8.76%** | **3.13%** |

---

## 4. Per-dataset scoring audit

Naming convention used below: **automatic** = deterministic rule with no model
judgment; **judge-based** = LLM semantic judgment under a frozen blinded protocol;
**human-adjudicated** = an explicit human decision record.

### 4.1 LongMemEval

| Property | Value |
|---|---|
| Scoring unit | one generated answer (6,666 rows in the full population) |
| Native evaluation criteria | benchmark-native answer/abstention semantics; official harness uses an LLM judge, so "native" here is *native-style*, not the upstream script |
| Implemented procedure | hierarchical, exclusive routing: normalized exact/native alternative, canonical scalar/yes-no/date equivalence (**automatic**); canonical abstention when the native source establishes answerability (**automatic**); otherwise **judge-based**. Semantic labels: `correct`, `incorrect`, `abstention_correct`, `abstention_incorrect`, `adjudication_required`; the two `*_correct` states project to binary correctness while raw labels are retained |
| Full-population routes | deterministic_native 853, deterministic_abstention 2,523, semantic_judge_required 3,286, invalid 4 |
| Formal-300 observed routes (450 LME observations) | deterministic_native 67, deterministic_abstention 163, semantic 220 |
| Judge disagreement (Formal-300 LME) | 218 dual-primary judged; 211 agree, 7 disagree (rate 0.9679); 9 Gemini escalation requests, 7 used for binary majority |
| Judge disagreement (B4 604-cohort LME) | 96 cases; 89 primary exact agreements, 6 binary-only agreements despite raw subtype disagreement, 0 true binary disagreements requiring the third judge, 1 persistent failure |
| Adjudication status | Formal-300: 2 observations unresolved (`primary_missing_or_unusable`); B4 cohort: 0 human-review-required, 1 human adjudication |
| Full native-protocol equivalence | **Not established.** Deterministic "native equivalence" is a documented conservative proxy, and the upstream LongMemEval harness itself is judge-based; the paper cannot claim byte-identical reproduction of the official evaluator |

### 4.2 Memora

| Property | Value |
|---|---|
| Scoring unit | native: one generated answer scored over its **atomic criteria**, aggregated by `FAMA = max(0, MPA − λ(1 − FAA))`; criteria split into `memory_presence` and `forgetting_absence` |
| Native evaluation criteria | per-answer criterion set released by the benchmark (15,960 answers → 252,336 criterion assignments: 109,326 presence, 143,010 absence) |
| Implemented procedure | one **judge-based** Gemini primary batch per answer carrying that answer's full criterion set; frozen OpenAI secondary risk sample; Anthropic third judgment on disagreement; strict structured parsing that fails closed; batch validation requires exact ID-set, count, no duplicates, no unknown IDs |
| Full-population routes | 15,955 semantic answers / 252,144 criteria; 192 criteria on invalid answers |
| Formal-300 observed | 450 Memora observations: Gemini primary direct 444, OpenAI secondary required 6, Anthropic escalation completed 1, human adjudication 1 |
| Judge disagreement | Formal-300: secondary agreement 4 / disagreement 2 (the 2 disagreements resolved via third judge and one human adjudication). B4 604-cohort: 491 primary agreements, 17 disagreements, all 17 resolved by Gemini majority |
| Adjudication status | Formal-300: 1 observation carried as `memora_third_pending`. B4 cohort: 0 human-review-required, 0 unresolved |
| Full native-protocol equivalence | **Partial.** Criterion-level decisions and the FAMA aggregation are preserved for the *native* metric. But the frozen GovMem scorer is an answer-level Boolean over extracted `content_data` facts, and Memora's native metric is per-question FAMA, so **no same-unit scorer comparison exists** (see 4.4) |

### 4.3 OAKS Novel

| Property | Value |
|---|---|
| Scoring unit | one generated answer against native option parity (6,720 validation rows) |
| Native evaluation criteria | benchmark-native option equality (deterministic) |
| Implemented procedure | **automatic**: native option comparison, no judgments |
| Result | 6,714 resolved, **6,714 agree, 0 disagree**, 6 ambiguous/unscored |
| Judge disagreement | not applicable |
| Adjudication status | none required |
| Full native-protocol equivalence | **Established for the frozen scorer on the resolved rows** (option parity), but the 6 ambiguous rows remain unscored and must not be counted as agreement |

OAKS is therefore the one dataset where the frozen scorer is demonstrably
native-equivalent — which is exactly why the Formal-300 excluded it.

### 4.4 The B4 deterministic-scorer validation (the actual B4 test)

`final_scorer_validation/scorer_validation_summary.json` compares the frozen
source-aware deterministic scorer against the independently adjudicated semantic
reference built in §1:

| Group | N | Agreement | Rate | Cohen's κ | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 604 | 274 | 45.36% | 0.114 | 19 | 311 |
| LongMemEval | 96 | 60 | 62.50% | 0.259 | 12 | 24 |
| Memora | 508 | 214 | 42.13% | 0.112 | 7 | 287 |

Wilson 95% CI on the overall rate: **[0.414, 0.494]**. By adjudication method:
primary exact agreement 264/580 (45.5%), third-judge majority 5/17 (29.4%),
human adjudication 1/1, binary-only-agreement-with-raw-disagreement 4/6.

LongMemEval disagreement root causes (36 cases): semantic paraphrase missed by the
deterministic rule 14, abstention-handling difference 18, normalization/native
alternative gap 2, deterministic false positive 2.

The **unit-alignment audit** then rules the Memora comparison inadmissible:

* The frozen scorer's unit is one answer; Memora's native unit is one atomic criterion.
* Neither direction is defensible from the frozen artifacts: aggregating criteria to
  an all-pass Boolean "would invent a new target", and recovering criterion-level
  predictions is impossible because no immutable field-to-criterion mapping exists.
* Only 4 of 42 unique answers even have matching scorer-field and memory-presence
  criterion counts.
* `valid_memora_n = 0`; the revised overall metric is not computed, because mixing
  LongMemEval answers with Memora criteria is prohibited.

**Net:** the only same-unit scorer-validity evidence that survives is LongMemEval,
N = 96, 62.5% agreement (κ = 0.259). Memora has no valid same-unit estimate at all.

---

## 5. What the Formal-300 actually supports

Formal-300 answers the question *"does the corruption effect survive when the
answers are re-scored with a native-style evaluator?"* It does **not** answer
*"is the frozen deterministic scorer valid?"* — it never runs that scorer, it
replaces it.

### 5.1 Claim A — limited: corruption degrades downstream task quality

**Partially supported, with stated limits.** Frozen final analysis
(`private/formal_300_validation_v1/final_analysis/formal_300_final_metrics.json`):

| Quantity | Value |
|---|---|
| Resolved matched triplets | **298 / 300** |
| Resolved / unresolved observations | 898 / 2 |
| Human-adjudicated observations | 1 |
| clean / corrupted / enforced correctness | 0.6903 / 0.3349 / 0.6617 |
| corruption (clean − corrupted) | **+35.54 pp [29.78, 41.12]** |
| recovery (enforced − corrupted) | **+32.67 pp [26.94, 38.36]** |
| restoration gap (enforced − clean) | −2.86 pp [−5.94, +0.12] |
| Block outcomes | corruption damage 160/298 (53.7%), recovery 153/298 (51.3%), full restoration 275/298 (92.3%) |
| Precommitted signal label | `strong_positive` |

Both corruption CIs exclude zero with no source- or provider-subgroup reversal, so
the *direction* of the effect is well-supported on this sample. It supports a
**limited** claim only, because the sample is a deterministic 2.4% subsample, OAKS
is absent from the semantic leg, and the title-level magnitudes are not reproduced
(§5.3).

### 5.2 Claim B — population-level semantic effect for all six providers

**Not supported.** Reasons, each independently sufficient:

1. Coverage is 300/12,480 provider–pair units (2.40%); no OAKS provider–pair units
   are included.
2. The selection is deterministic and provider-balanced by construction, so the
   sample proportions are design artifacts, not population proportions.
3. Per-provider effect estimates rest on n = 49–50 blocks, and the frozen analysis
   publishes no per-provider confidence interval, so per-provider precision is
   unquantified in the artifacts.
4. Bases are reused across providers, so the 300 blocks are not 300 independent
   content draws.
5. A paired block bootstrap over a non-probability sample yields an internal
   precision statement, not a population estimate.

The Formal-300 per-provider point estimates themselves already show the widths are
not credibly narrow: corruption ranges from +17.5 pp (deepseek) to +51.1 pp (openai).

### 5.3 Claim C — validation of the full frozen-scorer magnitudes

**Not supported.** Two independent reasons:

1. **Different instrument.** The Formal-300 uses `final-native-evaluation-v2`
   (semantic judges + native FAMA + deterministic native routing); the paper's
   headline numbers come from the frozen source-aware deterministic scorer. The
   Formal-300 cannot validate a scorer it does not run.
2. **Magnitudes do not reproduce.** Paper headline corruption macro −27.55 pp vs
   Formal-300 +35.54 pp — an **8.0 pp discrepancy**; recovery +27.51 pp vs +32.67 pp.
   Per-source: Formal-300 LongMemEval +40.5 pp / Memora +30.6 pp; both sit outside
   the paper's single −27.55 pp macro.

The one instrument that *was* compared to an independent reference (the frozen
scorer, §4.4) shows 45.4% agreement and 311 false negatives, i.e. the frozen-scorer
magnitudes are exactly what the B4 concern calls into question.

---

## 6. Missing coverage and the smallest defensible supplementary plan

### 6.1 What is missing

| Gap | Quantity |
|---|---|
| Provider–pair units never scored by the native-style evaluator | 12,180 / 12,480 (97.60%) |
| Eligible LME+Memora provider–pair units never scored | 9,291 / 9,591 (96.87%) |
| OAKS provider–pair units in the semantic leg | 2,880 (deliberate: native parity already established) |
| Memora same-unit scorer-validity units | **0** (508 cases are unit-inadmissible) |
| LongMemEval same-unit scorer-validity units | 96, Wilson 95% CI half-width **≈ 9.5 pp** at p = 0.625 |
| Full-cohort magnitude validation | entirely absent (plan exists: 22,727 calls, never executed) |

### 6.2 Sizing arithmetic (no assumed power — direct interval computation)

Wilson 95% half-widths for a proportion:

| Target half-width | Required n at p = 0.5 | Achieved half-width at p = 0.625 |
|---|---:|---:|
| ±10.0 pp | 97 | (n = 96 → ±9.5 pp) |
| ±7.5 pp | 171 | n = 200 → ±6.7 pp |
| ±5.0 pp | **385** | n = 384 → ±4.8 pp |
| ±4.0 pp | 601 | n = 500 → ±4.2 pp |

These are precision statements for a same-unit agreement rate; they are not power
calculations for the paper's effect estimates, and no budget is asserted.

### 6.3 Plan S — smallest defensible supplementary scoring (scorer-validity goal)

Targets claim C's evidence gap and the Memora unit void. Uses only existing model
responses.

| Field | Value |
|---|---|
| **Sampling unit** | one (provider, base, condition) answer for the scorer's unit; LongMemEval at answer level, Memora at answer level with its complete native criterion set |
| **Target strata** | 6 providers × 3 conditions × 2 datasets; Memora additionally balanced over `native_task_group` (activity_todos, content_email_writeup, content_project_proposal, preference_recommendation) |
| **Proposed units** | **384 LongMemEval answers and 384 Memora answers** (n chosen for ±5 pp at p = 0.5; includes topping up the existing 96 LongMemEval cases by 288) |
| **Scoring protocol** | frozen B4 blinding: two independent primaries from different provider families, third judge only on observed disagreement, strict fail-closed parsing, unresolved cases to blinded human review. LongMemEval keeps its five-way decision contract and binary projection; Memora keeps criterion-level `pass`/`fail` and native FAMA. The comparison endpoint (frozen scorer vs reference) must be predeclared and frozen **before** scoring |
| **Approximate requests** | LongMemEval 384 × 2 ≈ 768 (+ escalation at the observed 7/96 ≈ 7.3% rate ≈ 28) ; Memora 384 × 2 ≈ 768 (+ escalation at the observed 17/508 ≈ 3.3% rate ≈ 13) → **≈ 1,577 requests** for 768 cases, versus 1,232 requests for the existing 604-case B4 cohort (≈1.3×) |
| **Smaller variant (Plan S′)** | 171 per dataset → ±7.5 pp, ≈ 700 requests |
| **Expected limitations** | Resolves only the *scorer-validity* question; still cannot produce a population-level effect estimate (that requires Plan D). Adds no coverage of OAKS (already native-equivalent). Memora's answer-level endpoint still requires a predeclared projection between native FAMA and the scorer's Boolean, and that projection is itself a design decision, not a frozen artifact |

Honest alternative with **zero** API requests: Plan A — restrict the paper to
control-plane/state-transition evidence and drop downstream-correctness claims.
That is the smallest scientifically defensible option if Plan S is not run
(matching `paper_impact_audit/remediation_options.json`).

### 6.4 Plan D — if the population-level claim (B) is retained

Existing frozen plan `full-native-reevaluation-v1`: score all 22,626 existing
LongMemEval+Memora answers (3,286 LME semantic × 2 primaries + 15,955 Memora Gemini
primaries + 200 frozen Memora secondary risk) = **22,727 base requests**, excluding
result-dependent escalations; OAKS keeps its 6,720 native-parity rows. This is the
only plan of the three that would substantiate claim B.

---

## 7. Verified evidence, unresolved issues, recommendation

### 7.1 Verified in this audit

* Frozen cohort: 731 bases / 2,080 matched pairs / 4,891 conditions; six providers;
  Formal-300 is exactly 300 blocks, 900 observations, fingerprint
  `<FORMAL_BLOCK_FINGERPRINT>`, resampled from nothing.
* Coverage: 2.40% of provider–pair units; 0% of OAKS; 239 distinct bases reused
  across providers.
* Formal-300 final: 298/300 resolved triplets, corruption +35.54 pp [29.78, 41.12],
  recovery +32.67 pp [26.94, 38.36], restoration gap −2.86 pp [−5.94, +0.12],
  `strong_positive`.
* Provenance is separable and correctly labelled: 230/450 LongMemEval Formal-300
  observations are automatic, 670/900 observations across the sample are
  judge-based, and exactly 1 observation is human-adjudicated.
* B4 scorer validation: 45.36% overall agreement (κ = 0.114), LongMemEval 62.50%
  (n = 96), Memora 42.13% (n = 508, **unit-inadmissible**).
* OAKS native parity: 6,714/6,714 resolved rows agree; 6 rows unscored.
* The cheap-judge route is correctly quarantined: DeepSeek and Kimi failed the
  evaluator-selection gate, so the pilot cannot replace `final-native-evaluation-v2`.

### 7.2 Unresolved methodological issues

1. **Memora same-unit validity is absent.** Until a predeclared endpoint bridging
   native FAMA and the scorer's Boolean exists, no Memora scorer-validity number
   can be reported — and none should be.
2. **LongMemEval validity rests on 96 cases** (±9.5 pp) with a documented
   paraphrase/abstention failure mode; the deterministic route is a proxy, not the
   upstream judge-based harness.
3. **The Formal-300 and the headline numbers use different instruments**, and the
   effect magnitudes differ by ~8 pp. Neither value may be presented as validating
   the other.
4. **Formal-300 is not a probability sample.** Its intervals are internal precision
   statements only; the paper must not present them as population estimates.
5. **Per-provider CIs are not computed** in the frozen Formal-300 analysis, so no
   provider-specific claim can currently cite a confidence interval.
6. **Two Formal-300 observations remain unresolved** and are correctly excluded;
   the 6 OAKS ambiguous rows are correctly excluded from agreement.
7. **The formal plan's own concurrency warning** (no interprocess lock on the shared
   ledger) is a reproducibility caveat for any future run.

### 7.3 Is additional scoring necessary?

It depends strictly on which claims are retained:

* Keep only control-plane/state-transition claims → **no** new scoring needed.
* Retain a *limited* downstream-quality claim → the Formal-300 already suffices; no
  new scoring needed, but the claim must carry the coverage, confounding, and
  instrument caveats verbatim.
* Retain a *population-level* semantic effect for all six providers → Plan D is
  required (22,727 requests).
* Retain the *frozen-scorer magnitudes* → Plan S is the smallest sufficient step
  (≈ 1,570 requests), and it is still not sufficient without Plan D for coverage.

### 7.4 Recommendation

**CONDITIONAL GO** — retain the paper's main claims, subject to all of the
following being stated in the manuscript and the following work being done.

Conditions:

1. **Narrow the claims.** Keep the corruption/recovery direction as a
   *native-style, 300-block, two-dataset* result; explicitly state 2.40% provider–pair
   coverage, no OAKS in the semantic leg, and that the sample is deterministic and
   not a probability sample. Do not claim population-level effects for six providers
   on this evidence (claim B is unsupported).
2. **Separate the instruments.** Report the Formal-300 (native-style) and the frozen
   deterministic scorer as two distinct instruments; do not present one as
   validating the other, and disclose the ~8 pp magnitude difference.
3. **Report the scorer-validity evidence honestly and per source.** LongMemEval:
   62.50% agreement, κ = 0.259, n = 96. Memora: no valid same-unit estimate, with the
   unit-alignment reason stated. OAKS: 6,714/6,714 native option parity.
4. **Freeze the endpoint before any new scoring.** Publicly freeze the
   scorer-vs-reference endpoint (and, for Memora, the FAMA→Boolean projection) prior
   to running Plan S; otherwise the new numbers repeat the same defect.
5. **Do not cite the Formal-300 CIs as population precision**, and do not cite
   per-provider estimates without computing their intervals.

If conditions 1–3 cannot be met, the fallback is Plan A: restrict the paper to the
scorer-independent control-plane/state evidence (731 bases, 2,080 pairs, 12,480
comparisons, 96.25% admission change, 100% restoration), which remains fully
verified and unaffected by B4.

### 7.5 Files created / changed

* `B4_FINAL_EVIDENCE_GAP_AUDIT.md` (this document).
* `scripts/audit_b4_evidence_gap.py` (offline coverage/exclusion reproduction).

No frozen manifest, score file, metric, adjudication record, or paper result was
modified, and nothing was committed or pushed. No manuscript was edited.

### 7.6 Repository-location note

All required artifacts (Formal-300 execution records, B4 calibration and
frontier-adjudication records, the frozen v3 manifests, and the original per-provider
score files) live in the private experimental repository: the `govmem-demo` main
worktree plus its linked `govmem-public-release` worktree, which share a single
`.git`. This audit was produced in the linked worktree because that is where the
Formal-300 and B4 material physically resides; the aggregate-only public
reproducibility release contains none of the row-level inputs used here.
