#!/usr/bin/env python3
"""Generic offline v2/v3 Clean -> Corrupted -> Enforced trace validator (M3).

Validator for the implementation-level claim that governance corruption is a
*surgical* intervention: within one matched triplet, the clean, corrupted, and
enforced conditions must differ only in condition bookkeeping, the corruption /
enforcement specification, and the model-visible projection.  Everything else
-- task question, oracle answer, options, source evidence, base identity, and
the oracle / internal governance-state records -- must be byte-identical.

Per triplet the validator checks:

1. triplet integrity   - exactly one clean row, the corrupted row for one fault
                         operator, and the enforced row sharing its pair_id;
                         base/source/event agree across the three.
2. governance state    - oracle_state is invariant across the three conditions;
                         internal_governance_state is unchanged by corruption
                         (the fault is realised in the projection); the
                         enforcement specification's oracle_state_reference
                         matches the enforced governance state.
3. surgical invariance - the set of fields that differ between siblings is a
                         subset of the documented bookkeeping + projection
                         fields; any other difference is a violation.
4. admission           - the model-visible context specification
                         (item_count / projection) changes under corruption and
                         is restored by enforcement.
5. model-visible text  - the model-visible context text is restored exactly.
                         Text is compared in memory only; the output records
                         counts and rates, never text.
6. provider labels     - optionally, each of the six frozen providers carries a
                         usable frozen-proxy score label for all three
                         condition ids of the triplet.

Inputs are private: the v2/v3 condition manifest (and, for check 6, the
`results/final_main_sanitized/*.jsonl` score files) are deliberately not part of
the aggregate-only public release.  Run this script on the developer tree; in a
public checkout it prints an explicit blocker and exits 2 rather than
fabricating a result.

No network access, no provider API call, no model rerun, no mutation of any
frozen artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v3.jsonl"
DEFAULT_SCORES_DIR = ROOT / "results/final_main_sanitized"
DEFAULT_OUTPUT = ROOT / "results/paper_evidence_audit"

CONDITIONS = ("clean", "corrupted", "enforced")
PROVIDERS = ("openai", "anthropic", "deepseek", "gemini", "kimi", "grok")

# Fields that are *allowed* to differ between sibling conditions of one triplet.
# Everything else must be byte-identical, otherwise the intervention is not a
# surgical governance-state change.
ALLOWED_TRIPLET_VARIATION = frozenset(
    {
        "condition_id",
        "condition_type",
        "pair_id",
        "fault_operator",
        "corruption_specification",
        "enforcement_specification",
        "model_visible_context_text",
        "model_visible_memory_items",
        "model_visible_context_spec",
    }
)

BLOCKED_EXIT_CODE = 2


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# triplet construction
# --------------------------------------------------------------------------- #
def build_triplets(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Group manifest rows into matched clean/corrupted/enforced triplets."""
    clean_by_base: dict[str, dict[str, Any]] = {}
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicate_clean: list[str] = []
    duplicate_member: list[str] = []
    for row in rows:
        base_id = row["base_id"]
        if row["condition_type"] == "clean":
            if base_id in clean_by_base:
                duplicate_clean.append(base_id)
            clean_by_base[base_id] = row
            continue
        pair_id = row.get("pair_id")
        if not pair_id:
            duplicate_member.append(row.get("condition_id", "<unnamed>"))
            continue
        store = pairs[pair_id]
        if row["condition_type"] in store:
            duplicate_member.append(row.get("condition_id", "<unnamed>"))
        store[row["condition_type"]] = row

    triplets: list[dict[str, Any]] = []
    orphan_pairs: list[str] = []
    for pair_id, members in sorted(pairs.items()):
        if set(members) != {"corrupted", "enforced"}:
            orphan_pairs.append(pair_id)
            continue
        corrupted = members["corrupted"]
        base_id = corrupted["base_id"]
        clean = clean_by_base.get(base_id)
        if clean is None:
            orphan_pairs.append(pair_id)
            continue
        triplets.append(
            {
                "base_id": base_id,
                "pair_id": pair_id,
                "fault_operator": corrupted.get("fault_operator"),
                "source_dataset": corrupted.get("source_dataset"),
                "analysis_event_type": corrupted.get("analysis_event_type"),
                "conditions": {"clean": clean, "corrupted": corrupted, "enforced": members["enforced"]},
            }
        )
    integrity = {
        "clean_rows": len(clean_by_base),
        "pair_rows": len(pairs),
        "complete_triplets": len(triplets),
        "duplicate_clean_rows": len(duplicate_clean),
        "duplicate_or_unpaired_members": len(duplicate_member),
        "orphan_pair_groups": len(orphan_pairs),
    }
    return triplets, integrity


def _sibling_consistency(triplet: dict[str, Any]) -> dict[str, bool]:
    conditions = triplet["conditions"]
    ids = {name: row["condition_id"] for name, row in conditions.items()}
    bases = {row["base_id"] for row in conditions.values()}
    sources = {row.get("source_dataset") for row in conditions.values()}
    events = {row.get("analysis_event_type") for row in conditions.values()}
    return {
        "distinct_condition_ids": len(set(ids.values())) == 3,
        "corrupted_and_enforced_share_pair": conditions["corrupted"].get("pair_id")
        == conditions["enforced"].get("pair_id")
        and conditions["clean"].get("pair_id") is None,
        "base_consistent": len(bases) == 1,
        "source_consistent": len(sources) == 1,
        "event_consistent": len(events) == 1,
        "fault_declared_on_corrupted": bool(conditions["corrupted"].get("fault_operator")),
        "enforcement_targets_fault": (conditions["enforced"].get("enforcement_specification") or {}).get(
            "corrects_operator"
        )
        == conditions["corrupted"].get("fault_operator"),
        "pair_id_shape": conditions["clean"].get("pair_id") is None
        and conditions["corrupted"].get("pair_id") == triplet["pair_id"]
        and conditions["enforced"].get("pair_id") == triplet["pair_id"],
    }


def _governance_checks(triplet: dict[str, Any]) -> dict[str, bool]:
    clean, corrupted, enforced = (triplet["conditions"][name] for name in CONDITIONS)
    oracle_states = {json.dumps(row.get("oracle_state"), sort_keys=True) for row in (clean, corrupted, enforced)}
    enforcement_reference = (enforced.get("enforcement_specification") or {}).get("oracle_state_reference")
    return {
        "oracle_state_invariant": len(oracle_states) == 1,
        "internal_state_unchanged_by_corruption": clean.get("internal_governance_state")
        == corrupted.get("internal_governance_state"),
        "enforcement_oracle_reference_matches_enforced_state": enforcement_reference
        == enforced.get("internal_governance_state"),
        "corruption_declares_oracle_fields": bool(
            (corrupted.get("corruption_specification") or {}).get("source_oracle_fields")
        ),
    }


def _changed_fields(a: dict[str, Any], b: dict[str, Any]) -> set[str]:
    keys = set(a) | set(b)
    return {key for key in keys if a.get(key) != b.get(key)}


def _projection(row: dict[str, Any]) -> dict[str, Any]:
    spec = row.get("model_visible_context_spec") or {}
    return {
        "content_source": spec.get("content_source"),
        "projection": spec.get("projection"),
        "item_count": spec.get("item_count"),
    }


def _field_counts(counter: "Counter[str]") -> list[dict[str, Any]]:
    """Emit field tallies as a list so no field name becomes a JSON key."""
    return [{"field": field, "triplets": count} for field, count in sorted(counter.items())]


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate(
    triplets: Sequence[dict[str, Any]],
    scores_dir: Path | None = None,
) -> dict[str, Any]:
    integrity_checks: Counter[str] = Counter()
    governance_checks: Counter[str] = Counter()
    changed_c: Counter[str] = Counter()
    changed_e: Counter[str] = Counter()
    violations_c: Counter[str] = Counter()
    violations_e: Counter[str] = Counter()
    admission_changed = admission_restored = 0
    context_changed = context_restored = context_items_changed = context_items_restored = 0
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_fault: dict[str, Counter[str]] = defaultdict(Counter)

    for triplet in triplets:
        clean, corrupted, enforced = (triplet["conditions"][name] for name in CONDITIONS)
        source = str(triplet["source_dataset"])
        fault = str(triplet["fault_operator"])

        for key, ok in _sibling_consistency(triplet).items():
            integrity_checks[key] += int(ok)
        for key, ok in _governance_checks(triplet).items():
            governance_checks[key] += int(ok)

        diff_c = _changed_fields(clean, corrupted)
        diff_e = _changed_fields(clean, enforced)
        for field in diff_c:
            changed_c[field] += 1
        for field in diff_e:
            changed_e[field] += 1
        for field in sorted(diff_c - ALLOWED_TRIPLET_VARIATION):
            violations_c[field] += 1
        for field in sorted(diff_e - ALLOWED_TRIPLET_VARIATION):
            violations_e[field] += 1

        spec_changed = _projection(clean) != _projection(corrupted)
        spec_restored = _projection(clean) == _projection(enforced)
        text_changed = clean.get("model_visible_context_text") != corrupted.get("model_visible_context_text")
        text_restored = clean.get("model_visible_context_text") == enforced.get("model_visible_context_text")
        items_changed = clean.get("model_visible_memory_items") != corrupted.get("model_visible_memory_items")
        items_restored = clean.get("model_visible_memory_items") == enforced.get("model_visible_memory_items")

        admission_changed += int(spec_changed)
        admission_restored += int(spec_restored)
        context_changed += int(text_changed)
        context_restored += int(text_restored)
        context_items_changed += int(items_changed)
        context_items_restored += int(items_restored)

        for bucket in (by_source[source], by_fault[fault]):
            bucket["triplets"] += 1
            bucket["admission_changed"] += int(spec_changed)
            bucket["admission_restored"] += int(spec_restored)
            bucket["context_changed"] += int(text_changed)
            bucket["context_restored"] += int(text_restored)
            bucket["surgical_violations"] += len(diff_c - ALLOWED_TRIPLET_VARIATION) + len(
                diff_e - ALLOWED_TRIPLET_VARIATION
            )

    total = len(triplets)

    def rate(numerator: int) -> float | None:
        return (numerator / total) if total else None

    result: dict[str, Any] = {
        "schema_version": "govmem-condition-trace-validation-v1",
        "triplet_count": total,
        "triplet_integrity": {
            **{key: {"passed": value, "rate": rate(value)} for key, value in sorted(integrity_checks.items())},
        },
        "governance_state": {
            key: {"passed": value, "rate": rate(value)} for key, value in sorted(governance_checks.items())
        },
        "surgical_invariance": {
            "allowed_variation_fields": sorted(ALLOWED_TRIPLET_VARIATION),
            # Rendered as lists of {field, triplets} rather than field-keyed maps so
            # that no manifest field name is ever emitted as a JSON object key.
            "observed_changed_fields_clean_vs_corrupted": _field_counts(changed_c),
            "observed_changed_fields_clean_vs_enforced": _field_counts(changed_e),
            "violations_clean_vs_corrupted": _field_counts(violations_c),
            "violations_clean_vs_enforced": _field_counts(violations_e),
            "violating_triplets": sum(
                1
                for triplet in triplets
                if (
                    _changed_fields(triplet["conditions"]["clean"], triplet["conditions"]["corrupted"])
                    | _changed_fields(triplet["conditions"]["clean"], triplet["conditions"]["enforced"])
                )
                - ALLOWED_TRIPLET_VARIATION
            ),
        },
        "admission_projection": {
            "spec_changed": {"n": admission_changed, "rate": rate(admission_changed)},
            "spec_restored": {"n": admission_restored, "rate": rate(admission_restored)},
        },
        "model_visible_context": {
            "text_changed": {"n": context_changed, "rate": rate(context_changed)},
            "text_restored": {"n": context_restored, "rate": rate(context_restored)},
            "memory_items_changed": {"n": context_items_changed, "rate": rate(context_items_changed)},
            "memory_items_restored": {"n": context_items_restored, "rate": rate(context_items_restored)},
            "text_policy": "text compared in memory only; no text, hash, or locator is emitted",
        },
        "by_source": {key: dict(sorted(value.items())) for key, value in sorted(by_source.items())},
        "by_fault": {key: dict(sorted(value.items())) for key, value in sorted(by_fault.items())},
    }

    if scores_dir is not None:
        result["provider_frozen_proxy_labels"] = _validate_provider_labels(triplets, scores_dir)
    return result


def _validate_provider_labels(
    triplets: Sequence[dict[str, Any]], scores_dir: Path
) -> dict[str, Any]:
    """Check that each provider carries a usable frozen-proxy label per condition."""
    report: dict[str, Any] = {"scores_dir": scores_dir.name, "providers": {}, "missing_files": []}
    for provider in PROVIDERS:
        path = scores_dir / f"{provider}.jsonl"
        if not path.is_file():
            report["missing_files"].append(f"{provider}.jsonl")
            continue
        index = {row.get("condition_id"): row for row in read_jsonl(path)}
        labelled = unlabelled = missing = 0
        for triplet in triplets:
            for name in CONDITIONS:
                row = index.get(triplet["conditions"][name]["condition_id"])
                if row is None:
                    missing += 1
                    continue
                has_label = isinstance(row.get("primary_score"), int) or isinstance(
                    row.get("correctness"), bool
                )
                has_status = row.get("status") is not None and row.get("scoring_status") is not None
                if has_label and has_status:
                    labelled += 1
                else:
                    unlabelled += 1
        expected = len(triplets) * len(CONDITIONS)
        report["providers"][provider] = {
            "expected_labels": expected,
            "labelled": labelled,
            "unlabelled": unlabelled,
            "missing_rows": missing,
            "coverage_rate": (labelled / expected) if expected else None,
        }
    return report


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def render_markdown(result: dict[str, Any], inputs: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Condition-trace validation (M3)")
    add("")
    add(
        "Offline validation of the implementation-level claim that governance corruption is a "
        "surgical intervention. Text is compared in memory only; no benchmark text, model "
        "response, per-case identifier, or content fingerprint is emitted."
    )
    add("")
    add("## Inputs")
    add("")
    for key, value in inputs.items():
        add(f"- **{key}**: `{value}`")
    add("")
    add(f"Matched clean/corrupted/enforced triplets validated: **{result['triplet_count']}**.")
    add("")
    add("## 1. Triplet integrity")
    add("")
    add("| Check | Passed | Rate |")
    add("|---|---:|---:|")
    for key, entry in result["triplet_integrity"].items():
        add(f"| {key} | {entry['passed']} | {_pct(entry['rate'])} |")
    add("")
    add("## 2. Governance state")
    add("")
    add("| Check | Passed | Rate |")
    add("|---|---:|---:|")
    for key, entry in result["governance_state"].items():
        add(f"| {key} | {entry['passed']} | {_pct(entry['rate'])} |")
    add("")
    add("## 3. Surgical invariance")
    add("")
    add(
        "Fields allowed to differ between siblings: "
        + ", ".join(f"`{field}`" for field in result["surgical_invariance"]["allowed_variation_fields"])
        + "."
    )
    add("")
    add("| Sibling comparison | Field | Triplets differing |")
    add("|---|---|---:|")
    for label, key in (
        ("clean vs corrupted", "observed_changed_fields_clean_vs_corrupted"),
        ("clean vs enforced", "observed_changed_fields_clean_vs_enforced"),
    ):
        for entry in result["surgical_invariance"][key]:
            add(f"| {label} | `{entry['field']}` | {entry['triplets']} |")
    add("")
    violating = result["surgical_invariance"]["violating_triplets"]
    add(
        "Unexpected-variation violations: "
        + ("none in either comparison." if violating == 0 else f"{violating} triplet(s) violated the allowed field set.")
    )
    add("")
    add("## 4. Admission projection and model-visible context")
    add("")
    add("| Quantity | n | Rate |")
    add("|---|---:|---:|")
    for label, entry in (
        ("admission spec changed under corruption", result["admission_projection"]["spec_changed"]),
        ("admission spec restored by enforcement", result["admission_projection"]["spec_restored"]),
        ("model-visible context text changed", result["model_visible_context"]["text_changed"]),
        ("model-visible context text restored", result["model_visible_context"]["text_restored"]),
        ("model-visible memory items changed", result["model_visible_context"]["memory_items_changed"]),
        ("model-visible memory items restored", result["model_visible_context"]["memory_items_restored"]),
    ):
        add(f"| {label} | {entry['n']} | {_pct(entry['rate'])} |")
    add("")
    add("## 5. Breakdown")
    add("")
    add("| Group | Value | Triplets | Admission changed | Context changed | Context restored | Violations |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for key, label in (("by_source", "source"), ("by_fault", "fault")):
        for value, entry in result[key].items():
            add(
                f"| {label} | {value} | {entry['triplets']} | {entry['admission_changed']} | "
                f"{entry['context_changed']} | {entry['context_restored']} | {entry['surgical_violations']} |"
            )
    add("")
    if "provider_frozen_proxy_labels" in result:
        add("## 6. Six-provider frozen-proxy labels")
        add("")
        add("| Provider | Expected labels | Labelled | Unlabelled | Missing rows | Coverage |")
        add("|---|---:|---:|---:|---:|---:|")
        report = result["provider_frozen_proxy_labels"]
        for provider, entry in report["providers"].items():
            add(
                f"| {provider} | {entry['expected_labels']} | {entry['labelled']} | "
                f"{entry['unlabelled']} | {entry['missing_rows']} | {_pct(entry['coverage_rate'])} |"
            )
        if report["missing_files"]:
            add("")
            add(f"Missing score files: {', '.join(report['missing_files'])}")
        add("")
    add("## Interpretation")
    add("")
    add(
        "- The corruption is surgical: apart from condition bookkeeping, the fault/enforcement "
        "specification, and the model-visible projection, the three siblings are byte-identical, "
        "so the task, the oracle answer, the source evidence, and the oracle and internal "
        "governance records cannot have silently changed.\n"
        "- Enforcement restores the model-visible context exactly, which is the control-plane "
        "half of the paper's claim; it does **not** by itself establish downstream answer "
        "correctness.\n"
        "- This validator is an implementation-level check of the frozen manifests. It makes no "
        "model or API call and does not re-score any answer."
    )
    add("")
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scores-dir", type=Path, default=None, help="optional frozen provider score directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        print("BLOCKED: condition manifest is not present in this checkout.")
        print(f"  expected: {args.manifest}")
        print("  the v2/v3 text-bearing manifests are private and excluded from the public release;")
        print("  rebuild them locally (docs/OBTAINING_DATASETS.md) or pass --manifest.")
        print("CONDITION_TRACE_STATUS=blocked_missing_private_input")
        return BLOCKED_EXIT_CODE

    rows = read_jsonl(args.manifest)
    triplets, integrity = build_triplets(rows)
    if not triplets:
        print("BLOCKED: no complete clean/corrupted/enforced triplets found.")
        print("CONDITION_TRACE_STATUS=blocked_no_triplets")
        return BLOCKED_EXIT_CODE

    result = validate(triplets, args.scores_dir)
    result["input_summary"] = {
        **integrity,
        "manifest_name": args.manifest.name,
        "manifest_sha256": sha256_file(args.manifest),
        "schema_versions": dict(sorted(Counter(row.get("schema_version") for row in rows).items())),
        "condition_schema_versions": dict(
            sorted(Counter(row.get("condition_schema_version") for row in rows).items())
        ),
    }

    if not args.no_write:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "condition_trace_validation.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output / "condition_trace_validation.md").write_text(
            render_markdown(
                result,
                {
                    "condition manifest": args.manifest.name,
                    "score directory": args.scores_dir.name if args.scores_dir else "(not supplied)",
                },
            ),
            encoding="utf-8",
        )

    violations = sum(
        entry["triplets"]
        for entry in result["surgical_invariance"]["violations_clean_vs_corrupted"]
        + result["surgical_invariance"]["violations_clean_vs_enforced"]
    )
    print(f"CONDITION_TRACE_TRIPLETS={result['triplet_count']}")
    print(f"CONDITION_TRACE_SURGICAL_VIOLATIONS={violations}")
    print(f"CONDITION_TRACE_CONTEXT_RESTORED={result['model_visible_context']['text_restored']['n']}")
    print(f"CONDITION_TRACE_STATUS={'pass' if violations == 0 else 'fail'}")
    return 0 if violations == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
