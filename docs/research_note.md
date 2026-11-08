# GovMem: Warehouse-Style Memory Governance for Long-Term LLM Agents

## Motivation

Long-term memory for LLM agents needs more than retrieval. Stored memories can become stale, user preferences can change, memories can be deleted, and some memories may be private or unauthorized. Conflicting memories also need resolution. In that setting, naive vector retrieval can surface unsafe or outdated facts even when it returns text that looks relevant.

## Research Question

Can a warehouse-style memory layer with governance policies improve the reliability and safety of long-term memory QA compared with naive retrieval?

## System Overview

This demo uses a small deterministic pipeline:

- Memory warehouse schema: a structured record format for memory content, timestamps, validity windows, source, scope, confidence, and entity key
- Policy engine: filters deleted, expired, and unauthorized memories before retrieval
- Conflict resolution: selects among competing memories using validity, source rank, confidence, and recency
- Retriever: lexical scoring over the remaining memories
- Deterministic answer generator: extracts answers from retrieved memory text using a simple pattern, without calling an LLM
- Evaluation runner and trace reports: compute aggregate metrics and produce per-example traces to show why each method succeeded or failed

## Methods Compared

- `NoMemoryBaseline`: answers without using memory
- `NaiveMemoryBaseline`: retrieves memory without governance
- `WarehouseMemoryNoGovernance`: stores and retrieves from a structured warehouse, but does not enforce policy
- `GovMem`: warehouse plus governance policy engine and conflict resolution

`WarehouseMemoryNoGovernance` is important because it isolates whether improvements come from structured storage alone or from governance policies. That distinction matters: a warehouse can organize memory, but it does not by itself prevent stale or unsafe retrieval.

## Metrics

- `exact_match`: normalized exact answer match
- `token_f1`: token overlap between prediction and gold answer
- `safe_exact_match`: exact match plus safe retrieval for that example
- `policy_violation_count`: number of retrieved memories that were deleted, expired, or unauthorized
- `stale_memory_error_count`: number of retrieved memories that were stale or out of date
- `avg_retrieved_memories`: average number of memories retrieved per example

## Demo Results

| method | EM | F1 | safe_EM | avg_k | policy | stale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 0.125 | 0.125 | 0.125 | 0.00 | 0 | 0 |
| naive_memory | 0.500 | 0.625 | 0.375 | 2.00 | 3 | 2 |
| warehouse_no_governance | 0.500 | 0.625 | 0.375 | 2.00 | 3 | 2 |
| govmem | 0.750 | 0.812 | 0.750 | 1.25 | 0 | 0 |

Takeaway:

- Warehouse storage alone does not fix unsafe retrieval.
- Governance reduces unsafe and stale retrieval.
- Conflict resolution improves changed-preference examples.
- `safe_exact_match` distinguishes being merely correct from being correct using safe evidence.

## Example Trace

Changed preference example:

- Naive retrieval returns both an old preference and the current preference, then predicts the old one.
- GovMem resolves the conflict and predicts the current preference.

Stale, deleted, or private example:

- Naive retrieval surfaces unsafe memory.
- GovMem filters the unsafe record before retrieval.

## Connection to LongMemEval

This demo uses LongMemEval-style long-term memory QA examples. It is compatible with the idea of evaluating long-memory agents, but it is not yet a full LongMemEval benchmark. A real evaluation would replace the synthetic dataset with LongMemEval examples or LongMemEval-derived governance slices, then measure the same safety and answer-quality tradeoffs at larger scale.

The repository now also includes an initial adapter for LongMemEval-style session data, which converts simplified conversation JSONL into the warehouse schema used by the demo.

The repository now includes a deterministic governance slice generator that derives stress cases from LongMemEval-style converted examples. These slices are intended to test governance-specific behavior, not to replace official benchmark evaluation.

The repo now also supports an initial official LongMemEval cleaned adapter. Official data is used here for schema-compatible retrieval evaluation, while full QA evaluation still requires replacing the deterministic answer generator with an LLM or benchmark evaluator.

## Limitations

- synthetic small dataset
- deterministic answer generator
- lexical retrieval
- no LLM judge in the default offline path
- no real embeddings
- no large-scale statistical significance
- handcrafted governance labels
- The generated governance slices intentionally amplify policy-sensitive cases, so they are best viewed as a stress test rather than a representative distribution.

## Next Steps

- add a real LongMemEval loader
- add an embedding retriever baseline
- add an LLM answer generator
- add an LLM judge or human evaluation
- add governance-specific slices
- scale experiments across multiple models
- compare against memory agents or long-context baselines
- measure token cost and retrieval precision

## Suggested Paper Framing

Long-term memory benchmarks usually focus on whether information can be recalled. GovMem asks whether recalled information is also valid, authorized, fresh, and conflict-resolved.
