#!/usr/bin/env python3
"""Offline coverage/exclusion audit for the B4 Formal-300 scoring validation.

Read-only with respect to every frozen artifact.  It never calls a model or a
paid API, never regenerates or re-scores an answer, and prints only aggregate
counts (no question, answer, criterion, or provider-response text).

It reconstructs, from the frozen inputs only:

* the Formal-300 block/observation identities and their strata;
* the eligible candidate universe the sampler drew from (same code path as
  ``scripts/prepare_cheap_judge_pilot.py``);
* coverage of the full frozen v3 cohort by provider / dataset / event / fault.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.design_final_native_evaluation import (  # noqa: E402
    is_valid_answer_row,
    load_answer_paths,
    load_existing_answers,
)
from scripts.prepare_cheap_judge_pilot import (  # noqa: E402
    PROVIDERS,
    SOURCES,
    build_candidates,
    read_jsonl,
)

CONDITION_MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v3.jsonl"
BASE_MANIFEST = ROOT / "data/final_main_governance_base_manifest_v2.jsonl"
SELECTED_BLOCKS = ROOT / "results/cheap_judge_pilot_v1/selected_blocks.jsonl"
SELECTED_ANSWERS = ROOT / "results/cheap_judge_pilot_v1/selected_answers.jsonl"
FORMAL_BLOCKS = ROOT / "results/formal_300_validation_v1/selected_observations.jsonl"

SOURCE_LABELS = {"longmemeval": "LongMemEval", "memora": "Memora", "oaks_novel": "OAKS Novel"}
EVENTS = ("deletion", "temporal", "update")
FAULTS = ("missing", "stale", "flipped", "conflicting")


def pct(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{100.0 * numerator / denominator:.2f}%"


def load_frozen_cohort() -> dict[str, Any]:
    conditions = read_jsonl(CONDITION_MANIFEST)
    bases = read_jsonl(BASE_MANIFEST)
    pairs: dict[str, dict[str, Any]] = {}
    for row in conditions:
        if row.get("pair_id") and row["condition_type"] == "corrupted":
            pairs[row["pair_id"]] = {
                "base_id": row["base_id"],
                "source": row["source_dataset"],
                "event": row["analysis_event_type"],
                "fault": row["fault_operator"],
            }
    return {"conditions": conditions, "bases": bases, "pairs": pairs}


def main() -> int:
    cohort = load_frozen_cohort()
    full_pairs = cohort["pairs"]
    full_bases = {b["base_id"]: b for b in cohort["bases"]}
    provider_n = len(PROVIDERS)

    # ---- eligible candidate universe (same code path as the frozen sampler) --
    answer_paths = load_answer_paths(ROOT)
    lme_rows, memora_rows, _provenance = load_existing_answers(answer_paths)
    all_rows = lme_rows + memora_rows
    candidates = build_candidates(all_rows)

    eligible_by_provider_source: Counter[tuple[str, str]] = Counter()
    eligible_pairs_by_provider: Counter[str] = Counter()
    for cand in candidates:
        eligible_by_provider_source[(cand["provider"], cand["source"])] += 1
        eligible_pairs_by_provider[cand["provider"]] += 1

    # bases that have at least one eligible pair per (provider, source)
    bases_with_eligible = defaultdict(set)
    for cand in candidates:
        bases_with_eligible[(cand["provider"], cand["source"])].add(cand["base_id"])

    rows_per_provider = Counter(r["provider"] for r in all_rows if is_valid_answer_row(r))

    # ---- the locked 300 blocks --------------------------------------------- #
    blocks = read_jsonl(SELECTED_BLOCKS)
    answers = read_jsonl(SELECTED_ANSWERS)
    formal = read_jsonl(FORMAL_BLOCKS)
    block_by_id = {b["pilot_block_id"]: b for b in blocks}

    formal_provider = Counter(b["generating_provider"] for b in blocks)
    formal_source = Counter(b["source"] for b in blocks)
    formal_event = Counter(b["event_macro"] for b in blocks)
    formal_fault = Counter(b["fault_type"] for b in blocks)
    formal_provider_source = Counter((b["generating_provider"], b["source"]) for b in blocks)
    formal_provider_event = Counter((b["generating_provider"], b["event_macro"]) for b in blocks)
    formal_provider_fault = Counter((b["generating_provider"], b["fault_type"]) for b in blocks)
    formal_source_event = Counter((b["source"], b["event_macro"]) for b in blocks)
    formal_source_fault = Counter((b["source"], b["fault_type"]) for b in blocks)

    distinct_bases = {b["base_id"] for b in blocks}
    bases_per_provider = Counter(len({b["base_id"] for b in blocks if b["generating_provider"] == p}) for p in PROVIDERS)
    base_reuse = Counter(Counter(b["base_id"] for b in blocks).values())

    # ---- full-cohort aggregates -------------------------------------------- #
    full_pairs_by_source = Counter(p["source"] for p in full_pairs.values())
    full_pairs_by_event = Counter(p["event"] for p in full_pairs.values())
    full_pairs_by_fault = Counter(p["fault"] for p in full_pairs.values())
    full_pairs_source_event = Counter((p["source"], p["event"]) for p in full_pairs.values())
    full_pairs_source_fault = Counter((p["source"], p["fault"]) for p in full_pairs.values())
    full_bases_by_source = Counter(b["source_dataset"] for b in full_bases.values())
    full_bases_by_event = Counter(b["analysis_event_type"] for b in full_bases.values())

    def table(title: str, header: list[str], rows: list[list[str]]) -> None:
        print(f"\n### {title}\n")
        widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
        print("| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(header)) + " |")
        print("|" + "|".join("-" * (widths[i] + 2) for i in range(len(header))) + "|")
        for r in rows:
            print("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)) + " |")

    print("## Frozen cohort identity")
    print(f"- condition manifest pairs: {len(full_pairs)}")
    print(f"- base manifest bases: {len(full_bases)}")
    print(f"- providers: {provider_n} {list(PROVIDERS)}")
    print(f"- provider x pair universe: {len(full_pairs) * provider_n}")
    print(f"- provider x base universe: {len(full_bases) * provider_n}")
    print(f"- valid LME+Memora conditions per provider: {dict(rows_per_provider)}")

    print("\n## Formal-300 composition")
    print(f"- blocks: {len(blocks)}  observations: {len(answers)} (formal manifest: {len(formal)})")
    print(f"- distinct bases: {len(distinct_bases)}; block reuse per base: {dict(base_reuse)}")
    print(f"- per provider: {dict(formal_provider)}")
    print(f"- per source: {dict(formal_source)}")
    print(f"- per event: {dict(formal_event)}")
    print(f"- per fault: {dict(formal_fault)}")

    print("\n## Eligible candidate universe (offline recomputation)")
    rows = []
    for provider in PROVIDERS:
        for source in SOURCES:
            elig = eligible_by_provider_source[(provider, source)]
            elig_bases = len(bases_with_eligible[(provider, source)])
            sel = formal_provider_source[(provider, source)]
            rows.append([provider, source, str(elig_bases), str(elig), str(sel),
                         pct(sel, elig_bases) + " (base)", pct(sel, elig) + " (pair)"])
    table("Eligible vs selected per provider x source",
          ["provider", "source", "elig_bases", "elig_pairs", "selected", "sel/base%", "sel/pair%"], rows)
    tot_elig_bases = sum(len(bases_with_eligible[(p, s)]) for p in PROVIDERS for s in SOURCES)
    tot_elig_pairs = sum(eligible_by_provider_source[(p, s)] for p in PROVIDERS for s in SOURCES)
    print(f"\n- total eligible provider-base units: {tot_elig_bases}")
    print(f"- total eligible provider-pair units: {tot_elig_pairs}")
    print(f"- selected 300 -> {pct(300, tot_elig_bases)} of eligible provider-base units, "
          f"{pct(300, tot_elig_pairs)} of eligible provider-pair units")

    print("\n## Coverage vs full frozen v3 cohort (provider-replicated)")
    rows = []
    for provider in list(PROVIDERS) + ["ALL"]:
        for source in list(SOURCES) + ["oaks_novel"]:
            pass
    # by provider
    rows = []
    for provider in PROVIDERS:
        sel_pairs = formal_provider[provider]
        rows.append([provider, str(sel_pairs), str(len(full_pairs)), pct(sel_pairs, len(full_pairs)),
                     str(len(full_bases)), pct(sel_pairs, len(full_bases))])
    rows.append(["ALL (6 providers)", str(len(blocks)), str(len(full_pairs) * provider_n),
                 pct(len(blocks), len(full_pairs) * provider_n),
                 str(len(full_bases) * provider_n), pct(len(blocks), len(full_bases) * provider_n)])
    table("By provider (selected blocks vs one-provider pair / base universe)",
          ["provider", "formal300", "v3_pairs", "cov/pair", "v3_bases", "cov/base"], rows)

    print("\n## Coverage by dataset (provider-replicated denominators)")
    rows = []
    for source in ("longmemeval", "memora", "oaks_novel"):
        sel = formal_source[source]
        pair_units = full_pairs_by_source[source] * provider_n
        base_units = full_bases_by_source[source] * provider_n
        rows.append([SOURCE_LABELS[source], str(sel), str(pair_units), pct(sel, pair_units),
                     str(base_units), pct(sel, base_units)])
    rows.append(["ALL", str(len(blocks)), str(len(full_pairs) * provider_n),
                 pct(len(blocks), len(full_pairs) * provider_n),
                 str(len(full_bases) * provider_n), pct(len(blocks), len(full_bases) * provider_n)])
    table("By dataset", ["dataset", "formal300", "v3_pair_units", "cov/pair", "v3_base_units", "cov/base"], rows)

    print("\n## Coverage by event (provider-replicated denominators)")
    rows = []
    for event in EVENTS:
        sel = formal_event[event]
        pair_units = full_pairs_by_event[event] * provider_n
        base_units = full_bases_by_event[event] * provider_n
        rows.append([event, str(sel), str(pair_units), pct(sel, pair_units),
                     str(base_units), pct(sel, base_units)])
    table("By event", ["event", "formal300", "v3_pair_units", "cov/pair", "v3_base_units", "cov/base"], rows)

    print("\n## Coverage by fault operator (provider-replicated denominators)")
    rows = []
    for fault in FAULTS:
        sel = formal_fault[fault]
        pair_units = full_pairs_by_fault[fault] * provider_n
        rows.append([fault, str(sel), str(pair_units), pct(sel, pair_units)])
    table("By fault", ["fault", "formal300", "v3_pair_units", "cov/pair"], rows)

    print("\n## provider x dataset (provider-replicated denominators)")
    rows = []
    for provider in PROVIDERS:
        for source in ("longmemeval", "memora"):
            sel = formal_provider_source[(provider, source)]
            pair_units = full_pairs_by_source[source] * provider_n
            rows.append([provider, SOURCE_LABELS[source], str(sel), str(pair_units), pct(sel, pair_units)])
    table("provider x dataset", ["provider", "dataset", "formal300", "v3_pair_units", "cov"], rows)

    print("\n## provider x fault (provider-replicated denominators)")
    rows = []
    for provider in PROVIDERS:
        for fault in FAULTS:
            sel = formal_provider_fault[(provider, fault)]
            pair_units = full_pairs_by_fault[fault] * provider_n
            rows.append([provider, fault, str(sel), str(pair_units), pct(sel, pair_units)])
    table("provider x fault", ["provider", "fault", "formal300", "v3_pair_units", "cov"], rows)

    print("\n## Formal-300 source x event / source x fault")
    print("source x event:", dict(formal_source_event))
    print("source x fault:", dict(formal_source_fault))

    # ---- adjudication / resolution provenance ------------------------------ #
    obs_path = ROOT / "results/formal_300_validation_v1/analysis/resolved_observations.jsonl"
    blk_path = ROOT / "results/formal_300_validation_v1/analysis/resolved_blocks.jsonl"
    if obs_path.is_file() and blk_path.is_file():
        obs = read_jsonl(obs_path)
        blks = read_jsonl(blk_path)
        print("\n## Formal-300 final adjudicated analysis")
        print(f"- resolved observations: {len(obs)} (of 900 selected)")
        print(f"- unresolved observations: {900 - len(obs)}")
        print(f"- resolved matched triplets: {len(blks)} (of 300)")
        print("- resolution_method:", dict(Counter(o["resolution_method"] for o in obs)))
        print("- route:", dict(Counter(o["route"] for o in obs)))
        print("- observations by source:", dict(Counter(o["source"] for o in obs)))
        print("- observations by fault:", dict(Counter(o["fault_type"] for o in obs)))

    # ---- exclusion accounting ---------------------------------------------- #
    print("\n## Exclusion accounting (LME + Memora)")
    oaks_pair_units = full_pairs_by_source["oaks_novel"] * provider_n
    oaks_base_units = full_bases_by_source["oaks_novel"] * provider_n
    print(f"- OAKS provider-pair units excluded by protocol: {oaks_pair_units}")
    print(f"- OAKS provider-base units excluded by protocol: {oaks_base_units}")
    lme_mem_pair_units = (full_pairs_by_source["longmemeval"] + full_pairs_by_source["memora"]) * provider_n
    print(f"- LME+Memora provider-pair units in the cohort: {lme_mem_pair_units}")
    print(f"- ... eligible after complete-triplet validity filter: {tot_elig_pairs}")
    print(f"- ... of which selected: {len(blocks)}; not selected: {tot_elig_pairs - len(blocks)}")
    print(f"- provider-pair units dropped by triplet-validity filter: {lme_mem_pair_units - tot_elig_pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
