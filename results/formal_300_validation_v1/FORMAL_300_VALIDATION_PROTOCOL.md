# Formal 300-block validation protocol

Protocol `formal-300-validation-v1` applies the frozen `final-native-evaluation-v2` evaluator to exactly the 300 blocks fingerprinted `<FORMAL_BLOCK_FINGERPRINT>`. It performs no resampling and reuses existing answers only.

LongMemEval uses frozen deterministic/native routing where safe. Each of the 220 semantic observations receives independent blinded OpenAI `gpt-5.6-sol` and Anthropic `claude-sonnet-5` judgments; binary disagreement creates a result-dependent Gemini `gemini-3-flash-preview` escalation. Memora batches all native criteria for one answer through Gemini, preserving native FAMA. The intersection with the globally frozen 200-case risk sample creates 6 OpenAI secondary requests. Low-confidence/adjudication outputs create result-dependent secondary requests; criterion disagreement creates an Anthropic third judgment. Unresolved cases require human review.

Prompts contain no generating provider, condition, GovMem score, cheap-judge result, expected effect, paper result, or other judge output. Dynamic requests are materialized only from frozen triggers and retain the same private semantic payload. Provider ledgers/results are separate and single-writer. Completed slots skip; failed and started-only slots fail closed; no automatic retries.

Analysis uses complete matched clean/corrupted/enforced triplets and a fixed-seed paired bootstrap over provider/base blocks. LongMemEval contributes binary correctness; Memora contributes native per-answer FAMA. This formal plan excludes OAKS because its frozen native deterministic evaluator remains unchanged.
