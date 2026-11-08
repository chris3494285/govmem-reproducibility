#!/usr/bin/env python3
"""Independently verify a built public-release candidate.

Checks performed, all offline:

1. The candidate contains no ``.git`` directory or Git metadata.
2. Every file in ``MANIFEST.sha256`` exists with the exact recorded SHA-256.
3. Every file present on disk (excluding the manifest files themselves) is
   listed in the manifest.
4. No file matches a forbidden pattern from the allowlist's public-data audit
   (machine-specific absolute paths, credential-shaped tokens, benchmark
   answer/question text fields, request-identifier fields).

Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_public_release_candidate import (  # noqa: E402
    audit_text,
    compile_forbidden,
    exemptions_for,
    load_allowlist,
)

MANIFEST_FILES = {"MANIFEST.sha256", "FILE_MANIFEST.json", "FILE_MANIFEST.csv"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path("release/public_release_allowlist.json"),
    )
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    allowlist_path = args.allowlist
    if not allowlist_path.is_absolute():
        allowlist_path = Path.cwd() / allowlist_path
    spec = load_allowlist(allowlist_path)
    forbidden = compile_forbidden(spec["forbidden_patterns"])
    exemptions = spec.get("audit_exemptions", [])

    failures: list[str] = []

    if (candidate / ".git").exists():
        failures.append("candidate contains a .git directory")

    disk_files = {
        p.relative_to(candidate).as_posix()
        for p in candidate.rglob("*")
        if p.is_file() and p.relative_to(candidate).as_posix() not in MANIFEST_FILES
    }

    recorded: dict[str, str] = {}
    manifest_path = candidate / "MANIFEST.sha256"
    if not manifest_path.exists():
        failures.append("missing MANIFEST.sha256")
    else:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, rel = line.split("  ", 1)
            recorded[rel] = digest

    for rel, digest in recorded.items():
        target = candidate / rel
        if not target.exists():
            failures.append(f"manifest entry missing on disk: {rel}")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            failures.append(f"hash mismatch: {rel}")

    for rel in sorted(disk_files - set(recorded)):
        failures.append(f"unlisted file on disk: {rel}")

    violations: list[dict] = []
    for rel in sorted(disk_files):
        target = candidate / rel
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append({"path": rel, "rule": "non_utf8_binary", "example": "<binary>"})
            continue
        violations.extend(audit_text(rel, text, forbidden, exemptions_for(rel, exemptions)))

    for item in violations:
        failures.append(
            f"forbidden content: {item['path']}:{item['line']} "
            f"[{item['rule']}] (matched content redacted)"
        )

    print(f"candidate: {candidate}")
    print(f"files on disk: {len(disk_files)}")
    print(f"manifest entries: {len(recorded)}")
    print(f"content violations: {len(violations)}")
    if failures:
        print(f"FAIL: {len(failures)} problem(s)")
        for item in failures[:50]:
            print(f"  - {item}")
        return 1
    print("PASS: candidate is internally consistent and free of forbidden content")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
