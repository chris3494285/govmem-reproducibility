# Approved v3 selective rerun commands

All six identities pass `python scripts/audit_rerun_identity.py`. These are the only live commands approved by the identity audit. They are recorded here but were not executed.

Each command requires the provider-specific environment variable documented in `rerun_cost_estimate.json`, and runs exactly the 265 changed IDs against the v3 manifest. The shared runner performs the hard identity gate before any provider request.

```bash
python scripts/run_final_main_openai.py --model gpt-5.6-sol --manifest data/final_main_governance_condition_manifest_v3.jsonl --results-dir results/answer_leakage_repair/rerun_openai --condition-ids-file results/answer_leakage_repair/changed_condition_ids.txt --token-budget-config configs/final_main_token_budget_v2.json
python scripts/run_final_main_anthropic.py --model claude-sonnet-5 --manifest data/final_main_governance_condition_manifest_v3.jsonl --results-dir results/answer_leakage_repair/rerun_anthropic --condition-ids-file results/answer_leakage_repair/changed_condition_ids.txt --token-budget-config configs/final_main_token_budget_anthropic_repair_v2.json
python scripts/run_final_main_deepseek.py --model deepseek-v4-pro --manifest data/final_main_governance_condition_manifest_v3.jsonl --results-dir results/answer_leakage_repair/rerun_deepseek --condition-ids-file results/answer_leakage_repair/changed_condition_ids.txt --token-budget-config configs/final_main_token_budget_v2.json
python scripts/run_final_main_gemini.py --model gemini-3-flash-preview --manifest data/final_main_governance_condition_manifest_v3.jsonl --results-dir results/answer_leakage_repair/rerun_gemini --condition-ids-file results/answer_leakage_repair/changed_condition_ids.txt --token-budget-config configs/final_main_token_budget_gemini_flash_repair_v2.json
python scripts/run_final_main_kimi.py --model kimi-k2.6 --base-url https://api.moonshot.cn/v1 --thinking disabled --manifest data/final_main_governance_condition_manifest_v3.jsonl --results-dir results/answer_leakage_repair/rerun_kimi --condition-ids-file results/answer_leakage_repair/changed_condition_ids.txt --token-budget-config configs/final_main_token_budget_kimi_repair_v1.json
python scripts/run_final_main_grok.py --model grok-4.3 --base-url https://api.x.ai/v1 --reasoning-effort none --manifest data/final_main_governance_condition_manifest_v3.jsonl --results-dir results/answer_leakage_repair/rerun_grok --condition-ids-file results/answer_leakage_repair/changed_condition_ids.txt --token-budget-config configs/final_main_token_budget_grok_v1.json
```

Resume uses the same command with `--resume`. Validate each output with the corresponding merge command in `rerun_cost_estimate.json`; that merge excludes exactly the changed IDs from the historical sanitized baseline, then validates all 4,891 v3 IDs. Do not use one completed file as both independent analysis runs: primary metrics use one complete provider result, while replication statistics require a genuinely independent second run.
