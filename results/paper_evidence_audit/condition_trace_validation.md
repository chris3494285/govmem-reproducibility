# Condition-trace validation (M3)

Offline validation of the implementation-level claim that governance corruption is a surgical intervention. Text is compared in memory only; no benchmark text, model response, per-case identifier, or content fingerprint is emitted.

## Inputs

- **condition manifest**: `final_main_governance_condition_manifest_v3.jsonl`
- **score directory**: `final_main_sanitized`

Matched clean/corrupted/enforced triplets validated: **2080**.

## 1. Triplet integrity

| Check | Passed | Rate |
|---|---:|---:|
| base_consistent | 2080 | 100.00% |
| corrupted_and_enforced_share_pair | 2080 | 100.00% |
| distinct_condition_ids | 2080 | 100.00% |
| enforcement_targets_fault | 2080 | 100.00% |
| event_consistent | 2080 | 100.00% |
| fault_declared_on_corrupted | 2080 | 100.00% |
| pair_id_shape | 2080 | 100.00% |
| source_consistent | 2080 | 100.00% |

## 2. Governance state

| Check | Passed | Rate |
|---|---:|---:|
| corruption_declares_oracle_fields | 2080 | 100.00% |
| enforcement_oracle_reference_matches_enforced_state | 2080 | 100.00% |
| internal_state_unchanged_by_corruption | 2080 | 100.00% |
| oracle_state_invariant | 2080 | 100.00% |

## 3. Surgical invariance

Fields allowed to differ between siblings: `condition_id`, `condition_type`, `corruption_specification`, `enforcement_specification`, `fault_operator`, `model_visible_context_spec`, `model_visible_context_text`, `model_visible_memory_items`, `pair_id`.

| Sibling comparison | Field | Triplets differing |
|---|---|---:|
| clean vs corrupted | `condition_id` | 2080 |
| clean vs corrupted | `condition_type` | 2080 |
| clean vs corrupted | `corruption_specification` | 2080 |
| clean vs corrupted | `fault_operator` | 2080 |
| clean vs corrupted | `model_visible_context_spec` | 1832 |
| clean vs corrupted | `model_visible_context_text` | 2002 |
| clean vs corrupted | `model_visible_memory_items` | 2002 |
| clean vs corrupted | `pair_id` | 2080 |
| clean vs enforced | `condition_id` | 2080 |
| clean vs enforced | `condition_type` | 2080 |
| clean vs enforced | `corruption_specification` | 2080 |
| clean vs enforced | `enforcement_specification` | 2080 |
| clean vs enforced | `fault_operator` | 2080 |
| clean vs enforced | `pair_id` | 2080 |

Unexpected-variation violations: none in either comparison.

## 4. Admission projection and model-visible context

| Quantity | n | Rate |
|---|---:|---:|
| admission spec changed under corruption | 1832 | 88.08% |
| admission spec restored by enforcement | 2080 | 100.00% |
| model-visible context text changed | 2002 | 96.25% |
| model-visible context text restored | 2080 | 100.00% |
| model-visible memory items changed | 2002 | 96.25% |
| model-visible memory items restored | 2080 | 100.00% |

## 5. Breakdown

| Group | Value | Triplets | Admission changed | Context changed | Context restored | Violations |
|---|---|---:|---:|---:|---:|---:|
| source | longmemeval | 460 | 382 | 382 | 460 | 0 |
| source | memora | 1140 | 1105 | 1140 | 1140 | 0 |
| source | oaks_novel | 480 | 345 | 480 | 480 | 0 |
| fault | conflicting | 618 | 540 | 540 | 618 | 0 |
| fault | flipped | 380 | 345 | 380 | 380 | 0 |
| fault | missing | 731 | 731 | 731 | 731 | 0 |
| fault | stale | 351 | 216 | 351 | 351 | 0 |

## 6. Six-provider frozen-proxy labels

| Provider | Expected labels | Labelled | Unlabelled | Missing rows | Coverage |
|---|---:|---:|---:|---:|---:|
| openai | 6240 | 6240 | 0 | 0 | 100.00% |
| anthropic | 6240 | 6233 | 7 | 0 | 99.89% |
| deepseek | 6240 | 6240 | 0 | 0 | 100.00% |
| gemini | 6240 | 6240 | 0 | 0 | 100.00% |
| kimi | 6240 | 6240 | 0 | 0 | 100.00% |
| grok | 6240 | 6240 | 0 | 0 | 100.00% |

## Interpretation

- The corruption is surgical: apart from condition bookkeeping, the fault/enforcement specification, and the model-visible projection, the three siblings are byte-identical, so the task, the oracle answer, the source evidence, and the oracle and internal governance records cannot have silently changed.
- Enforcement restores the model-visible context exactly, which is the control-plane half of the paper's claim; it does **not** by itself establish downstream answer correctness.
- This validator is an implementation-level check of the frozen manifests. It makes no model or API call and does not re-score any answer.
