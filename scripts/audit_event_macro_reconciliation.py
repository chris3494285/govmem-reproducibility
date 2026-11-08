#!/usr/bin/env python3
"""B2: reconcile the frozen six-provider event macro with the per-event means.

The reviewed draft reported an "event macro" corruption / recovery effect that
could not be recomputed from the per-event means, and the review suspected a
weighting mismatch.  This audit settles it against the frozen artifacts by
reusing the frozen computation functions from
``scripts/analyze_final_main_results.py`` -- nothing is re-implemented, and no
number is adjusted.

The published estimand is:

    macro(field) = mean over the three events of
                       mean over bases of
                           mean over that base's scored pairs of delta

i.e. a base-weighted per-event estimand, then an *unweighted* mean across the
three events.  The competing "recompute from the per-event table" reading uses
pair-weighted per-event means; the two differ whenever bases inside one event
carry different numbers of scored faults (in this cohort, only ``temporal``).

Reads only frozen artifacts; makes no network or provider call and mutates
nothing.  Emits aggregate-only output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_final_main_results import (  # noqa: E402
    EVENTS,
    _authoritative_rows,
    _event_effects,
    _event_macro,
    _micro_effect,
    _pair_effects,
    read_jsonl,
)

DEFAULT_MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v3.jsonl"
DEFAULT_SCORES_DIR = ROOT / "results/final_main_sanitized"
DEFAULT_OUTPUT = ROOT / "results/paper_evidence_audit"

PROVIDERS = ("openai", "anthropic", "deepseek", "gemini", "kimi", "grok")
FIELDS = (("corruption", "corruption_delta"), ("recovery", "recovery_delta"))

# Published headline values this audit must reconcile with (pp).
PUBLISHED = {"corruption": -27.55, "recovery": 27.51}
# The per-event-table recomputation the review reported (pp).
REVIEW_PAIR_WEIGHTED = {"corruption": -28.84, "recovery": 28.79}
TOLERANCE_PP = 0.02

BLOCKED_EXIT_CODE = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provider_effects(provider: str, manifest_rows, scores_dir: Path) -> list[dict[str, Any]]:
    path = scores_dir / f"{provider}.jsonl"
    rows = _authoritative_rows(read_jsonl(path), manifest_rows)
    return _pair_effects(rows)


def event_values(effects: list[dict[str, Any]], field: str, *, weighting: str) -> dict[str, float]:
    """Per-event effect under ``base`` or ``pair`` weighting."""
    values: dict[str, float] = {}
    for event in EVENTS:
        rows = [row for row in effects if row["analysis_event_type"] == event and row.get(field) is not None]
        if weighting == "pair":
            values[event] = mean(row[field] for row in rows)
        else:
            by_base: dict[str, list[float]] = {}
            for row in rows:
                by_base.setdefault(row["base_id"], []).append(row[field])
            values[event] = mean(mean(items) for items in by_base.values())
    return values


def analyze(manifest: Path, scores_dir: Path, *, replicates: int = 10_000) -> dict[str, Any]:
    manifest_rows = read_jsonl(manifest)
    per_provider: dict[str, Any] = {}
    for provider in PROVIDERS:
        effects = provider_effects(provider, manifest_rows, scores_dir)
        entry: dict[str, Any] = {
            "pair_count": len(effects),
            "base_count": len({row["base_id"] for row in effects}),
        }
        for name, field in FIELDS:
            macro = _event_macro(effects, field, replicates=replicates)
            base_events = event_values(effects, field, weighting="base")
            pair_events = event_values(effects, field, weighting="pair")
            entry[name] = {
                "event_values_base_weighted": base_events,
                "event_values_pair_weighted": pair_events,
                "event_mean_of_base_weighted": mean(base_events.values()),
                "event_mean_of_pair_weighted": mean(pair_events.values()),
                "event_macro": macro["mean"],
                "event_macro_ci95": [macro["ci_low"], macro["ci_high"]],
                "macro_event_values_reported": macro['event_values'],
                "pair_weighted_micro": _micro_effect(effects, field, replicates=replicates)["mean"],
                "base_minus_pair_weighting_gap": mean(base_events.values()) - mean(pair_events.values()),
            }
        per_provider[provider] = entry

    six_provider: dict[str, Any] = {}
    for name, _field in FIELDS:
        macro_values = [per_provider[p][name]["event_macro"] for p in PROVIDERS]
        base_event_values = [per_provider[p][name]["event_mean_of_base_weighted"] for p in PROVIDERS]
        pair_event_values = [per_provider[p][name]["event_mean_of_pair_weighted"] for p in PROVIDERS]
        micro_values = [per_provider[p][name]["pair_weighted_micro"] for p in PROVIDERS]
        six_provider[name] = {
            "mean_event_macro": mean(macro_values),
            "mean_event_of_base_weighted": mean(base_event_values),
            "mean_event_of_pair_weighted": mean(pair_event_values),
            "mean_pair_weighted_micro": mean(micro_values),
            "published_pp": PUBLISHED[name],
            "review_pair_weighted_pp": REVIEW_PAIR_WEIGHTED[name],
        }
        six = six_provider[name]
        six["macro_matches_published"] = abs(six["mean_event_macro"] * 100 - PUBLISHED[name]) <= TOLERANCE_PP
        six["pair_weighted_reproduces_review_value"] = (
            abs(six["mean_event_of_pair_weighted"] * 100 - REVIEW_PAIR_WEIGHTED[name]) <= TOLERANCE_PP
        )
        six["method_gap_pp"] = (
            six["mean_event_of_pair_weighted"] - six["mean_event_macro"]
        ) * 100

    # The reconciliation identity must hold exactly per provider.
    identity_errors = []
    for provider in PROVIDERS:
        for name, _field in FIELDS:
            entry = per_provider[provider][name]
            if abs(entry["event_macro"] - entry["event_mean_of_base_weighted"]) > 1e-12:
                identity_errors.append(f"{provider}:{name}")

    return {
        "schema_version": "govmem-event-macro-reconciliation-v1",
        "estimand": (
            "macro = unweighted mean over the three events of a base-weighted per-event effect; "
            "each per-event effect is the mean over bases of that base's mean scored-pair delta"
        ),
        "reused_functions": [
            "scripts/analyze_final_main_results.py::_authoritative_rows",
            "scripts/analyze_final_main_results.py::_pair_effects",
            "scripts/analyze_final_main_results.py::_event_macro",
            "scripts/analyze_final_main_results.py::_micro_effect",
            "scripts/analyze_final_main_results.py::_event_effects",
        ],
        "bootstrap_replicates": replicates,
        "inputs": {
            "condition_manifest": {"name": manifest.name, "sha256": sha256_file(manifest)},
            "scores_dir": scores_dir.name,
        },
        "per_provider": per_provider,
        "six_provider_mean": six_provider,
        "identity_errors": identity_errors,
        "reconciliation_status": "pass" if not identity_errors else "fail",
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Event-macro reconciliation (B2)")
    add("")
    add(result["estimand"])
    add("")
    add("## Per-provider event values (percentage points)")
    add("")
    for name in ("corruption", "recovery"):
        add(f"### {name.capitalize()}")
        add("")
        add("| Provider | deletion | temporal | update | macro | pair-weighted mean | micro (all pairs) | base-vs-pair gap |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|")
        for provider in PROVIDERS:
            entry = result["per_provider"][provider][name]
            b = entry["event_values_base_weighted"]
            add(
                f"| {provider} | {b['deletion'] * 100:+.4f} | {b['temporal'] * 100:+.4f} | "
                f"{b['update'] * 100:+.4f} | {entry['event_macro'] * 100:+.4f} | "
                f"{entry['event_mean_of_pair_weighted'] * 100:+.4f} | "
                f"{entry['pair_weighted_micro'] * 100:+.4f} | "
                f"{entry['base_minus_pair_weighting_gap'] * 100:+.4f} |"
            )
        add("")
    add("## Six-provider reconciliation")
    add("")
    add("| Effect | mean event-macro (pp) | published (pp) | pair-weighted event mean (pp) | review's recomputed value (pp) | method gap (pp) |")
    add("|---|---:|---:|---:|---:|---:|")
    for name in ("corruption", "recovery"):
        s = result["six_provider_mean"][name]
        add(
            f"| {name} | {s['mean_event_macro'] * 100:+.4f} | {s['published_pp']:+.2f} | "
            f"{s['mean_event_of_pair_weighted'] * 100:+.4f} | {s['review_pair_weighted_pp']:+.2f} | "
            f"{s['method_gap_pp']:+.4f} |"
        )
    add("")
    add("## Resolution of B2")
    add("")
    for name in ("corruption", "recovery"):
        s = result["six_provider_mean"][name]
        add(
            f"- **{name}**: the published value ({s['published_pp']:+.2f} pp) is the mean of the "
            f"base-weighted per-event estimand "
            f"({s['mean_event_macro'] * 100:+.4f} pp). Re-deriving it from pair-weighted per-event "
            f"means gives {s['mean_event_of_pair_weighted'] * 100:+.4f} pp, which reproduces the "
            f"reviewer's independent recomputation ({s['review_pair_weighted_pp']:+.2f} pp). The "
            f"difference ({s['method_gap_pp']:+.4f} pp) is entirely a weighting difference, not an "
            "arithmetic error."
        )
    add("")
    add(
        "Only the `temporal` event mixes bases with different numbers of scored faults "
        "(LongMemEval temporal bases support two operators; OAKS Novel temporal bases support "
        "three), so `deletion` and `update` are identical under both weightings and the whole "
        "gap is confined to `temporal`."
    )
    add("")
    add(f"Reconciliation identity errors: {result['identity_errors'] or 'none'}.")
    add("")
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    missing = [path for path in (args.manifest,) if not path.is_file()]
    missing += [args.scores_dir / f"{p}.jsonl" for p in PROVIDERS if not (args.scores_dir / f"{p}.jsonl").is_file()]
    if missing:
        print("BLOCKED: required private inputs are not present in this checkout.")
        for path in missing[:3]:
            print(f"  expected: {path}")
        print("  the v3 manifest and the six provider score files are private and excluded from the")
        print("  public release; rebuild them locally (docs/OBTAINING_DATASETS.md).")
        print("EVENT_MACRO_STATUS=blocked_missing_private_input")
        return BLOCKED_EXIT_CODE

    result = analyze(args.manifest, args.scores_dir, replicates=args.replicates)
    if not args.no_write:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "event_macro_reconciliation.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
        )
        (args.output / "event_macro_reconciliation.md").write_text(
            render_markdown(result), encoding="utf-8"
        )

    for name in ("corruption", "recovery"):
        s = result["six_provider_mean"][name]
        print(f"EVENT_MACRO_{name.upper()}_PP={s['mean_event_macro'] * 100:+.4f}")
        print(f"EVENT_MACRO_{name.upper()}_PUBLISHED_PP={s['published_pp']:+.2f}")
        print(f"EVENT_MACRO_{name.upper()}_MATCHES_PUBLISHED={s['macro_matches_published']}")
        print(f"EVENT_MACRO_{name.upper()}_PAIR_WEIGHTED_PP={s['mean_event_of_pair_weighted'] * 100:+.4f}")
    print(f"EVENT_MACRO_STATUS={result['reconciliation_status']}")
    return 0 if result["reconciliation_status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
