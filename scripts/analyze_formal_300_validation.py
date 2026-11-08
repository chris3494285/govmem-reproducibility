#!/usr/bin/env python3
"""Offline paired analysis for the frozen formal 300-block validation."""

from __future__ import annotations

import argparse, json, random, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_cheap_judge_pilot import read_jsonl
from scripts.prepare_formal_300_validation import EVALUATOR, PROTOCOL, write_json, write_jsonl
from scripts.run_formal_300_validation import criterion_map, result_index

BOOTSTRAP_SEED = 20261010


def fama(decisions: dict[str, str], prompt: str) -> float | None:
    criteria = json.loads(prompt)["criteria"]
    kinds = {str(c["criterion_id"]): str(c["evaluation_type"]) for c in criteria}
    if set(decisions) != set(kinds) or any(v not in {"pass", "fail"} for v in decisions.values()): return None
    mp = [k for k, v in kinds.items() if v == "memory_presence"]; fa = [k for k, v in kinds.items() if v == "forgetting_absence"]
    mpa = sum(decisions[k] == "pass" for k in mp) / len(mp) if mp else 1.0
    faa = sum(decisions[k] == "pass" for k in fa) / len(fa) if fa else 1.0
    return max(0.0, mpa - (len(fa) / len(kinds)) * (1.0 - faa))


def lme_binary(row: dict[str, Any] | None) -> bool | None:
    return None if row is None else row.get("parsed_result", {}).get("exploratory_binary")


def resolve(observations: list[dict[str, Any]], deterministic: list[dict[str, Any]], payloads: dict[str, dict[str, Any]],
            results: dict[tuple[str, str], dict[str, Any]], plan: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    det = {r["observation_id"]: r for r in deterministic}; counters = Counter(); resolved = []
    static_secondary = {r["observation_id"] for r in plan if r["judge_slot"] == "memora_secondary_openai"}
    for obs in observations:
        oid = obs["observation_id"]
        if oid in det:
            score = float(bool(det[oid]["binary_correct"])); method = "native_deterministic"
        elif obs["source"] == "longmemeval":
            a = results.get((oid, "lme_primary_openai")); b = results.get((oid, "lme_primary_anthropic"))
            av, bv = lme_binary(a), lme_binary(b)
            if av is None or bv is None:
                score = None; method = "primary_missing_or_unusable"
            elif av == bv:
                score = float(av); method = "dual_primary_binary_agreement"; counters["lme_primary_agreement"] += 1
            else:
                counters["lme_primary_disagreement"] += 1
                third = lme_binary(results.get((oid, "lme_escalation_gemini")))
                if third is None:
                    score = None; method = "gemini_escalation_pending"
                else:
                    score = float(sum((av, bv, third)) >= 2); method = "gemini_binary_majority"; counters["lme_escalation_completed"] += 1
        else:
            payload = payloads[oid]; primary = results.get((oid, "memora_primary_gemini"))
            pm = criterion_map(primary) if primary else None
            decisions = primary.get("parsed_result", {}).get("decisions", []) if primary else []
            dynamic_risk = any(x.get("confidence") == "low" or x.get("decision") == "adjudication_required" for x in decisions)
            needs_secondary = oid in static_secondary or dynamic_risk
            if pm is None:
                score = None; method = "memora_primary_missing"
            elif not needs_secondary:
                score = fama(pm, payload["prompt"]); method = "memora_gemini_primary"
            else:
                secondary = results.get((oid, "memora_secondary_openai")) or results.get((oid, "memora_dynamic_secondary_openai"))
                sm = criterion_map(secondary) if secondary else None
                if sm is None:
                    score = None; method = "memora_secondary_pending"
                elif pm == sm and all(v in {"pass", "fail"} for v in pm.values()):
                    score = fama(pm, payload["prompt"]); method = "memora_primary_secondary_agreement"; counters["memora_secondary_agreement"] += 1
                else:
                    counters["memora_secondary_disagreement"] += 1
                    third = results.get((oid, "memora_escalation_anthropic")); tm = criterion_map(third) if third else None
                    if tm is None or set(pm) != set(sm) or set(pm) != set(tm):
                        score = None; method = "memora_third_pending"
                    else:
                        majority = {}
                        for cid in pm:
                            votes = [pm[cid], sm[cid], tm[cid]]; valid = [v for v in votes if v in {"pass", "fail"}]
                            if valid.count("pass") >= 2: majority[cid] = "pass"
                            elif valid.count("fail") >= 2: majority[cid] = "fail"
                            else: majority[cid] = "adjudication_required"
                        score = fama(majority, payload["prompt"]); method = "memora_third_majority"
                        if score is not None: counters["memora_escalation_completed"] += 1
        if score is None: counters["unresolved_observations"] += 1
        resolved.append({**obs, "formal_score": score, "resolution_method": method})
    return resolved, counters


def metrics(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    if not blocks: return {"n": 0, "clean_accuracy": None, "corrupted_accuracy": None, "enforced_accuracy": None,
                           "clean_minus_corrupted": None, "enforced_minus_corrupted": None, "enforced_minus_clean": None}
    values = {c: statistics.mean(b["scores"][c] for b in blocks) for c in ("clean", "corrupted", "enforced")}
    return {"n": len(blocks), **{f"{k}_accuracy": v for k, v in values.items()},
            "clean_minus_corrupted": values["clean"] - values["corrupted"],
            "enforced_minus_corrupted": values["enforced"] - values["corrupted"],
            "enforced_minus_clean": values["enforced"] - values["clean"]}


def bootstrap(blocks: list[dict[str, Any]], iterations: int = 10_000) -> dict[str, Any]:
    if not blocks: return {k: None for k in ("clean_minus_corrupted", "enforced_minus_corrupted", "enforced_minus_clean")}
    rng = random.Random(BOOTSTRAP_SEED); samples = defaultdict(list)
    for _ in range(iterations):
        m = metrics([blocks[rng.randrange(len(blocks))] for _ in blocks])
        for k in ("clean_minus_corrupted", "enforced_minus_corrupted", "enforced_minus_clean"): samples[k].append(m[k])
    def pct(v: list[float], q: float) -> float:
        x = sorted(v); p = (len(x)-1)*q; lo = int(p); hi = min(lo+1,len(x)-1); return x[lo]+(x[hi]-x[lo])*(p-lo)
    return {k: [pct(v,.025), pct(v,.975)] for k,v in samples.items()}


def analyze(observations: list[dict[str, Any]], deterministic: list[dict[str, Any]], payloads: dict[str, dict[str, Any]],
            results: dict[tuple[str, str], dict[str, Any]], plan: list[dict[str, Any]], iterations: int = 10_000) -> dict[str, Any]:
    resolved, counters = resolve(observations, deterministic, payloads, results, plan)
    grouped = defaultdict(list)
    for r in resolved: grouped[r["pilot_block_id"]].append(r)
    blocks=[]
    for bid, rows in grouped.items():
        if len(rows)!=3 or {r["condition"] for r in rows}!={"clean","corrupted","enforced"}: raise ValueError("matched block drift")
        by={r["condition"]:r for r in rows}
        if all(by[c]["formal_score"] is not None for c in by):
            first=rows[0]; blocks.append({"pilot_block_id":bid,"source":first["source"],"generating_provider":first["generating_provider"],"fault_type":first["fault_type"],"scores":{c:by[c]["formal_score"] for c in by}})
    overall=metrics(blocks); overall["paired_bootstrap_95_ci"]=bootstrap(blocks,iterations)
    def split(field): return {v:metrics([b for b in blocks if str(b[field])==v]) for v in sorted({str(b[field]) for b in blocks})}
    by_source,by_provider,by_fault=split("source"),split("generating_provider"),split("fault_type")
    ci=overall["paired_bootstrap_95_ci"]; reversals=sum(x["n"] and (x["clean_minus_corrupted"]<0 or x["enforced_minus_corrupted"]<0) for table in (by_source,by_provider) for x in table.values())
    if len(blocks)<210: signal="inconclusive"
    elif ci["clean_minus_corrupted"][0]>0 and ci["enforced_minus_corrupted"][0]>0 and reversals==0 and abs(overall["enforced_minus_clean"])<abs(overall["clean_minus_corrupted"]): signal="strong_positive"
    elif overall["clean_minus_corrupted"]>0 and overall["enforced_minus_corrupted"]>0: signal="mixed"
    elif abs(overall["clean_minus_corrupted"])<.02 and abs(overall["enforced_minus_corrupted"])<.02: signal="weak"
    else: signal="mixed"
    return {"protocol":PROTOCOL,"evaluator":EVALUATOR,"total_blocks":300,
            "resolved_matched_triplets":len(blocks),"unresolved_observations":counters["unresolved_observations"],
            "overall":overall,"by_source":by_source,"by_generating_provider":by_provider,"by_fault_type":by_fault,
            "evaluator_counts":dict(counters),"formal_signal":signal,"bootstrap":{"seed":BOOTSTRAP_SEED,"iterations":iterations,"unit":"matched provider/base block"},
            "api_calls_performed":0,"resolved_observations":resolved,"resolved_blocks":blocks}


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output",type=Path,default=Path("results/formal_300_validation_v1/analysis")); p.add_argument("--iterations",type=int,default=10000)
    p.add_argument("--private",type=Path,default=Path("results/b4_scoring_validation/private/formal_300_validation_v1")); p.add_argument("--public",type=Path,default=Path("results/formal_300_validation_v1")); a=p.parse_args(argv)
    observations=read_jsonl(a.public/"selected_observations.jsonl"); deterministic=read_jsonl(a.public/"deterministic_results.jsonl"); plan=read_jsonl(a.public/"execution_plan.jsonl")
    dynamic=a.private/"dynamic_execution_plan.jsonl"; plan += read_jsonl(dynamic) if dynamic.exists() else []
    payloads={r["observation_id"]:r for r in read_jsonl(a.private/"semantic_payloads.jsonl")}
    result=analyze(observations,deterministic,payloads,result_index(a.private),plan,a.iterations); resolved=result.pop("resolved_observations"); blocks=result.pop("resolved_blocks")
    a.output.mkdir(parents=True,exist_ok=True); write_json(a.output/"formal_validation_summary.json",result); write_jsonl(a.output/"resolved_observations.jsonl",resolved); write_jsonl(a.output/"resolved_blocks.jsonl",blocks)
    (a.output/"FORMAL_300_VALIDATION_REPORT.md").write_text(f"# Formal 300 validation\n\nResolved matched triplets: {result['resolved_matched_triplets']}/300. Signal: `{result['formal_signal']}`. No API calls were made by analysis.\n",encoding="utf-8")
    print(f"RESOLVED_MATCHED_TRIPLETS={result['resolved_matched_triplets']}"); print(f"UNRESOLVED_OBSERVATIONS={result['unresolved_observations']}"); print(f"FORMAL_SIGNAL={result['formal_signal']}"); print("API_CALLS_PERFORMED=0"); return 0


if __name__=="__main__": raise SystemExit(main())
