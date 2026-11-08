# CLEAN-A / CLEAN-B stochasticity control — preflight (PHASE 1)

Offline only. No generations, no judge calls, no changes to frozen formal-300 artifacts.

## Sample

- Blocks: 60 (30 LongMemEval, 30 Memora).
- Per generating provider: {'anthropic': 10, 'deepseek': 10, 'gemini': 10, 'kimi': 10, 'openai': 10, 'xai': 10}.
- Seed: 20261013 (deterministic stratified selection).
- Duplicate formal-300 blocks: 0.
- Unique clean condition IDs: 56.

## Eligibility and non-conditioning on correctness

- Clean observations in the frozen sample: 300; resolved: 300; deterministic-route (excluded by route): 65; eligible semantic-resolved: 235.
- Selection uses only the frozen block/condition identifiers, the resolution *status*, and the frozen semantic route. The correctness value (True/False) is never read during selection; `clean_A_correctness` is joined in the manifest only after selection (flag: True).
- Blocks whose clean-A observation was scored by the frozen native-deterministic route are excluded so that CLEAN-A and CLEAN-B use the identical evaluator route; this exclusion is by route, not by correctness.
- CLEAN-B input verification: 54 of 60 blocks reconstruct the frozen generation prompt hash exactly; 6 blocks are fail-closed because the v3-corrected prompt text is not stored in the repository (only its hash). Generation is blocked for those slots.

## Frozen-input verification per block

Each manifest row stores the frozen clean condition ID, the model-visible-context hash, the generation prompt hash, the provider runner, and the inference configuration. CLEAN-B must reproduce the same `model_visible_context_sha256` and `prompt_sha256` before a generation call is allowed.

## Expected calls

- Generation calls: 60 by provider {'anthropic': 10, 'deepseek': 10, 'gemini': 10, 'kimi': 10, 'openai': 10, 'xai': 10}.
- Judge calls: base 90, expected 91, max 180.
- Expected generation input tokens (frozen CLEAN-A): 17345.
- Expected generation output tokens (frozen CLEAN-A): 6898.
- Pricing: configs/deepseek_v4_flash_pricing_v1.json has null price fields; no usable local pricing metadata, so dollar costs are not invented.

## Selected blocks

| repeat_block_id | provider | source | clean_A_correctness | evaluator route |
| --- | --- | --- | --- | --- |
| clean-repeat-v1::openai::longmemeval::01 | openai | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::openai::longmemeval::02 | openai | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::openai::longmemeval::03 | openai | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::openai::longmemeval::04 | openai | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::openai::longmemeval::05 | openai | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::openai::memora::06 | openai | memora | 0.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::openai::memora::07 | openai | memora | 1.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::openai::memora::08 | openai | memora | 1.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::openai::memora::09 | openai | memora | 1.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::openai::memora::10 | openai | memora | 1.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::anthropic::longmemeval::11 | anthropic | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::anthropic::longmemeval::12 | anthropic | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::anthropic::longmemeval::13 | anthropic | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::anthropic::longmemeval::14 | anthropic | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::anthropic::longmemeval::15 | anthropic | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::anthropic::memora::16 | anthropic | memora | 1.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::anthropic::memora::17 | anthropic | memora | 0.5 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::anthropic::memora::18 | anthropic | memora | 1.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::anthropic::memora::19 | anthropic | memora | 1.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::anthropic::memora::20 | anthropic | memora | 1.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::gemini::longmemeval::21 | gemini | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::gemini::longmemeval::22 | gemini | longmemeval | 0.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::gemini::longmemeval::23 | gemini | longmemeval | 0.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::gemini::longmemeval::24 | gemini | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::gemini::longmemeval::25 | gemini | longmemeval | 0.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::gemini::memora::26 | gemini | memora | 0.16666666666666669 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::gemini::memora::27 | gemini | memora | 1.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::gemini::memora::28 | gemini | memora | 1.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::gemini::memora::29 | gemini | memora | 1.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::gemini::memora::30 | gemini | memora | 1.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::kimi::longmemeval::31 | kimi | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::kimi::longmemeval::32 | kimi | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::kimi::longmemeval::33 | kimi | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::kimi::longmemeval::34 | kimi | longmemeval | 0.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::kimi::longmemeval::35 | kimi | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::kimi::memora::36 | kimi | memora | 0.3108448928121059 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::kimi::memora::37 | kimi | memora | 0.9090909090909091 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::kimi::memora::38 | kimi | memora | 1.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::kimi::memora::39 | kimi | memora | 0.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::kimi::memora::40 | kimi | memora | 1.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::xai::longmemeval::41 | xai | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::xai::longmemeval::42 | xai | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::xai::longmemeval::43 | xai | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::xai::longmemeval::44 | xai | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::xai::longmemeval::45 | xai | longmemeval | 0.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::xai::memora::46 | xai | memora | 1.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::xai::memora::47 | xai | memora | 0.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::xai::memora::48 | xai | memora | 0.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::xai::memora::49 | xai | memora | 0.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::xai::memora::50 | xai | memora | 0.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::deepseek::longmemeval::51 | deepseek | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::deepseek::longmemeval::52 | deepseek | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::deepseek::longmemeval::53 | deepseek | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::deepseek::longmemeval::54 | deepseek | longmemeval | 1.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::deepseek::longmemeval::55 | deepseek | longmemeval | 0.0 | LME dual-primary + Gemini escalation |
| clean-repeat-v1::deepseek::memora::56 | deepseek | memora | 0.55 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::deepseek::memora::57 | deepseek | memora | 0.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::deepseek::memora::58 | deepseek | memora | 1.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::deepseek::memora::59 | deepseek | memora | 0.0 | Memora Gemini primary + secondary/escalation |
| clean-repeat-v1::deepseek::memora::60 | deepseek | memora | 1.0 | Memora Gemini primary + secondary/escalation |
