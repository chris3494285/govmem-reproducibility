#!/usr/bin/env python3
"""Independent technical and content audit of a built JSS public package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_public_release_candidate import MANIFEST_FILES  # noqa: E402

REQUIRED_FILES = (
    "README.md",
    "ARTIFACTS.md",
    "THIRD_PARTY_DATASETS.md",
    "LICENSE",
    "LICENSE_SCOPE.md",
    "LICENSE_APPROVAL.md",
    "MATERIALS_REDISTRIBUTION_NOTICE.md",
    "CITATION.cff",
    "OAKS_DERIVED_ARTIFACTS.md",
    "JSS_PUBLIC_RELEASE_BUILD_REPORT.md",
    "JSS_PUBLIC_RELEASE_REMEDIATION_REPORT.md",
    "JSS_PUBLIC_RELEASE_FINAL_SIGNOFF.md",
    "docs/JSS_PUBLIC_RESULTS.md",
    "results/jss_public_aggregate_v1/rq1_observability_v2.json",
    "results/jss_public_aggregate_v1/scoring_consistency_sensitivity.json",
    "results/formal_300_subgroup_v1/bootstrap_comparison.json",
    "results/final_main_cross_provider_summary_v3/cross_provider_summary.json",
    "third_party/licenses/Adobe-Research-License-OAKS-Novel.txt",
    "third_party/licenses/Apache-2.0-Memora.txt",
    "third_party/licenses/MIT-LongMemEval.txt",
)
LICENSE_HASHES = {
    "third_party/licenses/Adobe-Research-License-OAKS-Novel.txt": (
        "d9a8d2d8348caaab20299ef6441854cce1312dc29f67923a94f2804826751739"
    ),
    "third_party/licenses/Apache-2.0-Memora.txt": (
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    ),
}
FORBIDDEN_NAMES = re.compile(
    r"(?:^|/)(?:private|judging)/|"
    r"(?:^|/)(?:\.env(?:\.|$)|generation_results\.jsonl|"
    r"judge_results\.jsonl|judge_execution_ledger\.jsonl|"
    r"semantic_payloads\.jsonl|.*adjudication.*\.jsonl)$"
)
SOURCE_IDENTIFIER = re.compile(
    r"(?:pilot-block|formal-300-slot|clean-repeat-judge|pilot-observation)::[0-9a-f]{16,}|"
    r"longmemeval::[0-9a-f]{6,}|"
    r"memora::(?:weekly|monthly|quarterly|yearly|daily)::|"
    r"oaks_novel::P[0-9]|memconflict::[0-9a-f]{8}-|gate"
    r"mem::"
)
SECRET_PATTERNS = (
    re.compile(r"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(r"/Users/[A-Za-z]|/home/[A-Za-z]"),
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_claims(root: Path) -> dict[str, Any]:
    rq1 = _load(root / "results/jss_public_aggregate_v1/rq1_observability_v2.json")
    scoring = _load(
        root / "results/jss_public_aggregate_v1/scoring_consistency_sensitivity.json"
    )
    bootstrap = _load(root / "results/formal_300_subgroup_v1/bootstrap_comparison.json")
    providers = _load(
        root / "results/final_main_cross_provider_summary_v3/cross_provider_summary.json"
    )
    claims = {
        "base_examples": rq1["cohort"]["bases"],
        "supported_pairs": rq1["cohort"]["matched_fault_pairs"],
        "conditions": rq1["cohort"]["condition_rows"],
        "providers": len(providers["models"]),
        "provider_replications": rq1["cohort"]["provider_replications"],
        "resolved_formal_triplets": rq1["downstream_output_observability"][
            "native_style"
        ]["resolved_blocks"],
        "resolved_distinct_bases": bootstrap["distinct_base_definitions"][
            "resolved_298_blocks"
        ],
        "bootstrap_iterations": bootstrap["cluster_bootstrap"]["iterations"],
        "bootstrap_seed": bootstrap["cluster_bootstrap"]["seed"],
        "identical_response_triplets": scoring["counts"]["overall"][
            "response_identity_unchanged"
        ],
        "identical_response_score_changes": scoring["counts"]["overall"][
            "response_identity_unchanged_score_changed"
        ],
    }
    expected = {
        "base_examples": 731,
        "supported_pairs": 2080,
        "conditions": 4891,
        "providers": 6,
        "provider_replications": 12480,
        "resolved_formal_triplets": 298,
        "resolved_distinct_bases": 238,
        "bootstrap_iterations": 10000,
        "bootstrap_seed": 20261010,
        "identical_response_triplets": 64,
        "identical_response_score_changes": 1,
    }
    mismatches = {
        key: {"expected": value, "actual": claims[key]}
        for key, value in expected.items()
        if claims[key] != value
    }
    if mismatches:
        raise ValueError(f"manuscript aggregate mismatch: {mismatches}")
    return claims


def validate_oaks_inventory(root: Path) -> dict[str, Any]:
    inventory_path = root / "OAKS_DERIVED_ARTIFACTS.md"
    text = inventory_path.read_text(encoding="utf-8")
    documented = set(re.findall(r"`((?:data|results)/[^`]+)`", text))
    observed: set[str] = set()
    for path in sorted((root / "data").rglob("*")) + sorted((root / "results").rglob("*")):
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "oaks" in content.lower():
            observed.add(path.relative_to(root).as_posix())
    return {
        "documented": len(documented),
        "observed": len(observed),
        "missing": sorted(observed - documented),
        "extra": sorted(documented - observed),
    }


def technical_audit(root: Path) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    files = sorted(path for path in root.rglob("*") if path.is_file())
    payload_files = [
        path
        for path in files
        if path.relative_to(root).as_posix() not in MANIFEST_FILES
    ]
    if (root / ".git").exists():
        failures.append("Git metadata is present")
    if any(path.is_symlink() for path in files):
        failures.append("symbolic link is present")
    for rel in REQUIRED_FILES:
        if not (root / rel).is_file():
            failures.append(f"required file missing: {rel}")
    for rel, expected in LICENSE_HASHES.items():
        path = root / rel
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"third-party license hash mismatch: {rel}")

    for path in files:
        rel = path.relative_to(root).as_posix()
        if FORBIDDEN_NAMES.search(rel):
            failures.append(f"forbidden private filename: {rel}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"unexpected binary file: {rel}")
            continue
        if rel != "release/aggregate_only_release_allowlist.json":
            match = SOURCE_IDENTIFIER.search(text)
            if match:
                failures.append(
                    f"source-derived identifier pattern in {rel} "
                    "(matched content redacted)"
                )
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                failures.append(
                    f"sensitive pattern in {rel} (matched content redacted)"
                )

    manifest_path = root / "MANIFEST.sha256"
    manifest_entries = 0
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, rel = line.split("  ", 1)
            manifest_entries += 1
            path = root / rel
            if not path.is_file():
                failures.append(f"manifest entry missing: {rel}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                failures.append(f"manifest hash mismatch: {rel}")
        if manifest_entries != len(payload_files):
            failures.append(
                f"manifest count mismatch: entries={manifest_entries}, files={len(payload_files)}"
            )
    else:
        failures.append("MANIFEST.sha256 is absent")

    try:
        claims = validate_claims(root)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        failures.append(str(exc))
        claims = {}
    try:
        oaks_inventory = validate_oaks_inventory(root)
        if oaks_inventory["missing"] or oaks_inventory["extra"]:
            failures.append(
                "OAKS-derived artifact inventory does not exactly match staged data/results"
            )
    except (FileNotFoundError, OSError) as exc:
        failures.append(str(exc))
        oaks_inventory = {}
    approval_text = (
        (root / "LICENSE_APPROVAL.md").read_text(encoding="utf-8")
        if (root / "LICENSE_APPROVAL.md").is_file()
        else ""
    ).lower()
    employer_confirmation = (
        "no employer, university, funder, or contractor owns or restricts"
        in approval_text
    )
    return failures, {
        "payload_files": len(payload_files),
        "manifest_entries": manifest_entries,
        "claims": claims,
        "oaks_inventory": oaks_inventory,
        "third_party_license_hashes_verified": not any(
            item.startswith("third-party license hash mismatch") for item in failures
        ),
        "govmem_license_present": any(
            (root / name).is_file()
            for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")
        ),
        "license_scope_present": (root / "LICENSE_SCOPE.md").is_file(),
        "author_approval_present": (root / "LICENSE_APPROVAL.md").is_file(),
        "materials_notice_present": (
            root / "MATERIALS_REDISTRIBUTION_NOTICE.md"
        ).is_file(),
        "employer_ip_confirmation_recorded": employer_confirmation,
        "citations_present": (root / "CITATION.cff").is_file(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--technical-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    failures, details = technical_audit(root)
    print(json.dumps(details, indent=2, sort_keys=True))
    if failures:
        print(f"JSS_TECHNICAL_AUDIT_STATUS=FAIL ({len(failures)} findings)")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("JSS_TECHNICAL_AUDIT_STATUS=PASS")
    if args.technical_only:
        return 0
    if not details["govmem_license_present"]:
        print("JSS_LICENSE_STATUS=BLOCKED (GovMem redistribution license not selected)")
        return 2
    if not details["license_scope_present"]:
        print("JSS_LICENSE_STATUS=BLOCKED (GovMem license scope is not documented)")
        return 2
    if not details["author_approval_present"]:
        print("JSS_LICENSE_STATUS=BLOCKED (coauthor approval is not recorded)")
        return 2
    if not details["materials_notice_present"]:
        print("JSS_LICENSE_STATUS=BLOCKED (materials redistribution notice is absent)")
        return 2
    if not details["citations_present"]:
        print("JSS_CITATION_STATUS=BLOCKED (CITATION.cff is absent)")
        return 2
    if not details["employer_ip_confirmation_recorded"]:
        print(
            "JSS_LICENSE_STATUS=BLOCKED "
            "(employer/institutional ownership confirmation is not recorded)"
        )
        return 2
    print("JSS_LICENSE_STATUS=REVIEWED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
