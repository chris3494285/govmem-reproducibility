#!/usr/bin/env python3
"""Build a licensing-safe pseudonymous variant of the public score files.

Motivation
----------
The default public score files in ``results/final_main_sanitized/`` keep
benchmark-derived identifiers (``base_id``/``condition_id``, which embed source
example ids) and per-row SHA-256 fingerprints of licensed text
(``model_visible_context_sha256``/``prompt_sha256``). Whether those derived
identifiers and fingerprints may be redistributed is unconfirmed.

This command produces the proposed fallback ("tier B") release:

* ``base_id``/``condition_id``/``pair_id`` are replaced by release-local,
  sequential pseudonyms assigned by a **published deterministic rule** (sort the
  distinct original ids and number them). A researcher who has licensed and
  rebuilt the manifests can reproduce the same ordering and recover the join
  exactly; the public artifact alone leaks no benchmark identifier.
* per-row licensed-text fingerprints are dropped.
* every non-derived scoring/accounting field is retained, so the probability
  that an aggregate statistic can be recomputed is unchanged.

No network access, no provider calls, no experiments. Writes only new files.
The optional ``--private-mapping`` table is for the authors' local use and must
NOT be published (it contains the original identifiers).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROVIDERS = ("openai", "anthropic", "deepseek", "gemini", "kimi", "grok")

# Derived identifiers replaced by pseudonyms.
ID_FIELDS = ("base_id", "condition_id", "pair_id")
# Licensed-text fingerprints removed entirely from the tier-B release.
FINGERPRINT_FIELDS = ("model_visible_context_sha256", "prompt_sha256")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def build(input_dir: Path, output_dir: Path, private_mapping: Path | None) -> dict:
    per_provider = {provider: _read_jsonl(input_dir / f"{provider}.jsonl") for provider in PROVIDERS}

    # Published deterministic pseudonymisation rule: sort distinct original ids
    # lexicographically and assign zero-padded sequential labels.
    base_ids = sorted({row["base_id"] for rows in per_provider.values() for row in rows})
    condition_ids = sorted({row["condition_id"] for rows in per_provider.values() for row in rows})
    pair_ids = sorted({str(row.get("pair_id")) for rows in per_provider.values() for row in rows})
    base_map = {value: f"b{index:04d}" for index, value in enumerate(base_ids)}
    cond_map = {value: f"c{index:05d}" for index, value in enumerate(condition_ids)}
    pair_map = {value: f"p{index:05d}" for index, value in enumerate(pair_ids)}

    output_dir.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for provider, rows in per_provider.items():
        renamed: list[dict] = []
        for row in rows:
            clean = {key: value for key, value in row.items() if key not in FINGERPRINT_FIELDS}
            if "base_id" in clean:
                clean["base_id"] = base_map[row["base_id"]]
            if "condition_id" in clean:
                clean["condition_id"] = cond_map[row["condition_id"]]
            if clean.get("pair_id") is not None:
                clean["pair_id"] = pair_map[str(row["pair_id"])]
            renamed.append(clean)
        path = output_dir / f"{provider}.jsonl"
        _write_jsonl(path, renamed)
        digests[provider] = hashlib.sha256(path.read_bytes()).hexdigest()

    if private_mapping is not None:
        private_mapping.parent.mkdir(parents=True, exist_ok=True)
        private_mapping.write_text(
            json.dumps(
                {
                    "warning": "PRIVATE — contains original benchmark-derived identifiers; never publish",
                    "base_id": {pseudonym: original for original, pseudonym in base_map.items()},
                    "condition_id": {pseudonym: original for original, pseudonym in cond_map.items()},
                    "pair_id": {pseudonym: original for original, pseudonym in pair_map.items()},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    manifest = {
        "schema_version": "final-main-pseudonymous-scores-v1",
        "tier": "B-licensing-safe-fallback",
        "source_dir": str(input_dir),
        "pseudonymisation_rule": (
            "distinct original identifiers sorted lexicographically and assigned "
            "sequential labels (base b0001..., condition c00001..., pair p00001...)"
        ),
        "replaced_fields": list(ID_FIELDS),
        "dropped_fields": list(FINGERPRINT_FIELDS),
        "retained_row_identity": False,
        "rows_per_provider": {provider: len(rows) for provider, rows in per_provider.items()},
        "provider_file_sha256": digests,
        "reproducibility_impact": [
            "Aggregate statistics (counts, effects, confidence intervals) remain fully recomputable.",
            "Row-level join to a rebuilt manifest is reproducible only by a party holding the licensed data and applying the published rule.",
            "Per-row context/prompt fingerprint verification against a rebuilt manifest is not possible.",
        ],
    }
    (output_dir / "pseudonymous_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/final_main_sanitized"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_main_tier_b"))
    parser.add_argument("--private-mapping", type=Path, help="local-only id mapping; never publish")
    args = parser.parse_args()
    manifest = build(args.input_dir, args.output_dir, args.private_mapping)
    print(json.dumps({k: manifest[k] for k in ("tier", "rows_per_provider", "replaced_fields", "dropped_fields")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
