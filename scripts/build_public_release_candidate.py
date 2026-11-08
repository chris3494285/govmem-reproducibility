#!/usr/bin/env python3
"""Build a public release candidate from an explicit allowlist.

This script never copies the repository wholesale. It resolves the
``include`` globs from ``release/public_release_allowlist.json`` against the
working tree, subtracts the ``exclude`` globs, applies the small documented
text-sanitisation rules, verifies that no copied file contains a forbidden
pattern, and only then writes the candidate plus a SHA-256 manifest.

The build fails closed: if any selected file still contains a forbidden
pattern (machine-specific absolute path, credential-shaped token, benchmark
answer/question text field, or request-identifier field), nothing is copied
and the manifest is not written.

No network access, no provider API calls, no Git mutation.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".cache",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
}
SKIP_FILE_NAMES = {".DS_Store", "Thumbs.db"}


def load_allowlist(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_includes(repo_root: Path, include_patterns: list[str]) -> set[str]:
    selected: set[str] = set()
    for pattern in include_patterns:
        matches = glob.glob(str(repo_root / pattern), recursive=True)
        for match in matches:
            candidate = Path(match)
            if not candidate.is_file():
                continue
            rel = candidate.relative_to(repo_root).as_posix()
            parts = rel.split("/")
            if any(part in SKIP_DIR_NAMES for part in parts):
                continue
            if parts[-1] in SKIP_FILE_NAMES:
                continue
            if rel.endswith(".pyc"):
                continue
            selected.add(rel)
    return selected


def apply_excludes(selected: set[str], exclude_patterns: list[str]) -> set[str]:
    kept: set[str] = set()
    for rel in selected:
        if any(fnmatch.fnmatchcase(rel, pat) for pat in exclude_patterns):
            continue
        kept.add(rel)
    return kept


def compile_forbidden(forbidden: list[dict]) -> list[dict]:
    compiled = []
    for entry in forbidden:
        compiled.append(
            {
                "name": entry["name"],
                "regex": re.compile(entry["regex"]),
                "extensions": set(entry["extensions"]),
            }
        )
    return compiled


def sanitize_text(text: str, rules: list[dict]) -> tuple[str, list[str]]:
    applied: list[str] = []
    for rule in rules:
        if rule["pattern"] in text:
            text = text.replace(rule["pattern"], rule["replacement"])
            applied.append(rule["name"])
    return text, applied


def audit_text(
    rel: str,
    text: str,
    forbidden: list[dict],
    exempt_rules: set[str],
) -> list[dict]:
    ext = os.path.splitext(rel)[1]
    violations: list[dict] = []
    for entry in forbidden:
        if entry["name"] in exempt_rules:
            continue
        if "*" not in entry["extensions"] and ext not in entry["extensions"]:
            continue
        match = entry["regex"].search(text)
        if match:
            violations.append(
                {
                    "path": rel,
                    "rule": entry["name"],
                    "line": text.count("\n", 0, match.start()) + 1,
                }
            )
    return violations


def exemptions_for(rel: str, exemptions: list[dict]) -> set[str]:
    rules: set[str] = set()
    for entry in exemptions:
        if fnmatch.fnmatchcase(rel, entry["path"]):
            rules.update(entry["rules"])
    return rules


def load_id_mapping(path: Path | None) -> dict[str, str]:
    """Load the private original-id -> pseudonym map (inverted for lookups)."""
    if path is None or not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for kind in ("base_id", "condition_id", "pair_id"):
        for pseudonym, original in data.get(kind, {}).items():
            mapping[str(original)] = pseudonym
    return mapping


def _reserialize(line_or_doc: str, is_jsonl: bool, drop_keys: set[str]) -> str | None:
    """Return a rewritten document with dropped keys, or None if unchanged."""

    def walk(node: object) -> bool:
        changed = False
        if isinstance(node, dict):
            for key in list(node):
                if key in drop_keys:
                    node.pop(key)
                    changed = True
                else:
                    changed |= walk(node[key])
        elif isinstance(node, list):
            for item in node:
                changed |= walk(item)
        return changed

    if is_jsonl:
        rows = [line for line in line_or_doc.splitlines() if line.strip()]
        changed = False
        out: list[str] = []
        for line in rows:
            obj = json.loads(line)
            if walk(obj):
                out.append(json.dumps(obj, ensure_ascii=False, sort_keys=True))
                changed = True
            else:
                out.append(line)
        return ("\n".join(out) + "\n") if changed else None
    obj = json.loads(line_or_doc)
    if not walk(obj):
        return None
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def apply_json_field_redactions(staged: dict[str, dict], spec: dict) -> dict:
    """Recursively remove explicitly reviewed fields from JSON/JSONL files."""
    drop_keys = set(spec.get("drop_json_fields", []))
    if not drop_keys:
        return {"files_redacted": 0, "fields_removed": 0}
    files_redacted = 0
    fields_removed = 0
    for rel, item in staged.items():
        ext = os.path.splitext(rel)[1]
        if ext not in (".json", ".jsonl"):
            continue
        text = item["data"].decode("utf-8")
        if not any(f'"{key}"' in text for key in drop_keys):
            continue
        before = text
        rewritten = _reserialize(text, ext == ".jsonl", drop_keys)
        if rewritten is None:
            continue
        item["data"] = rewritten.encode("utf-8")
        files_redacted += 1
        for key in drop_keys:
            fields_removed += before.count(f'"{key}"') - rewritten.count(f'"{key}"')
    return {"files_redacted": files_redacted, "fields_removed": fields_removed}


def _trim_token(token: str) -> str:
    return token.rstrip(".,;:)")


def apply_identifier_pseudonymization(
    staged: dict[str, dict],
    spec: dict,
    known_ids: dict[str, str],
) -> dict:
    """Rewrite source-derived identifiers and drop content-fingerprint fields.

    Two passes: first drop fingerprint fields from JSON/JSONL that contain them,
    then replace known source ids with their Tier-B pseudonyms and any remaining
    dataset-prefixed token with a deterministic release-local label.
    """
    extensions = set(spec["extensions"])
    drop_keys = set(spec.get("drop_json_fields", []))
    pattern = re.compile(spec["token_regex"])

    # Pass 1: drop content-fingerprint fields.
    dropped = 0
    for rel, item in staged.items():
        ext = os.path.splitext(rel)[1]
        if ext not in extensions or ext not in (".json", ".jsonl"):
            continue
        text = item["data"].decode("utf-8")
        if not any(f'"{key}"' in text for key in drop_keys):
            continue
        rewritten = _reserialize(text, ext == ".jsonl", drop_keys)
        if rewritten is not None:
            item["data"] = rewritten.encode("utf-8")
            dropped += 1

    # Pass 2: identifier pseudonymization in a single regex pass.
    unknown: set[str] = set()
    for rel, item in staged.items():
        if os.path.splitext(rel)[1] not in extensions:
            continue
        for match in pattern.finditer(item["data"].decode("utf-8")):
            token = _trim_token(match.group(0))
            if token and token not in known_ids:
                unknown.add(token)
    unknown_map = {token: f"x{index:05d}" for index, token in enumerate(sorted(unknown))}

    lookup = {**unknown_map, **known_ids}

    def _substitute(match: re.Match) -> str:
        raw = match.group(0)
        token = _trim_token(raw)
        if not token:
            return raw
        pseudonym = lookup.get(token)
        if pseudonym is None:
            return raw
        return pseudonym + raw[len(token):]

    replaced = 0
    for rel, item in staged.items():
        if os.path.splitext(rel)[1] not in extensions:
            continue
        text = item["data"].decode("utf-8")
        rewritten = pattern.sub(_substitute, text)
        if rewritten != text:
            item["data"] = rewritten.encode("utf-8")
            replaced += 1
    return {"files_with_fields_dropped": dropped, "files_with_ids_replaced": replaced,
            "unknown_id_tokens": len(unknown_map), "known_id_tokens": len(known_ids)}


def category_for(rel: str) -> str:
    if rel.startswith("govmem/"):
        return "package/govmem"
    if rel.startswith("eval/"):
        return "package/eval"
    if rel.startswith("baselines/"):
        return "package/baselines"
    if rel.startswith("scripts/"):
        return "scripts"
    if rel.startswith("configs/"):
        return "configs"
    if rel.startswith("tests/"):
        return "tests"
    if rel.startswith("docs/"):
        return "docs"
    if rel.startswith("third_party/"):
        return "third_party"
    if rel.startswith("data/"):
        return "frozen_metadata"
    if rel.startswith("results/final_main_sanitized/"):
        return "sanitized_scores"
    if rel.startswith("results/answer_leakage_repair/"):
        return "v3_analysis"
    if rel.startswith("results/stochasticity_control/"):
        return "clean_repeat"
    if rel.startswith("results/formal_300_validation_v1/"):
        return "formal_300"
    if rel.startswith("results/b4_scoring_validation/"):
        return "b4_scoring"
    if rel.startswith("results/final_main_cross_provider_summary_v3/"):
        return "v3_analysis"
    if rel.startswith("results/"):
        return "results_other"
    return "root"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path("release/public_release_allowlist.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/govmem-public-release-v3-candidate"),
    )
    parser.add_argument("--force", action="store_true", help="overwrite non-empty output dir")
    parser.add_argument("--dry-run", action="store_true", help="audit only, copy nothing")
    parser.add_argument(
        "--id-mapping",
        type=Path,
        help="private original->pseudonym mapping used for Tier-B consistency (never copied into the release)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    allowlist_path = (repo_root / args.allowlist) if not args.allowlist.is_absolute() else args.allowlist
    spec = load_allowlist(allowlist_path)
    if not spec.get("include"):
        raise SystemExit("allowlist must contain at least one include pattern")

    output = args.output.resolve()
    if output == repo_root or repo_root in output.parents:
        raise SystemExit("output must be outside the experiment repository")

    selected = resolve_includes(repo_root, spec["include"])
    selected = apply_excludes(selected, spec["exclude"])
    forbidden = compile_forbidden(spec["forbidden_patterns"])
    sanitize_rules = spec["sanitize_rules"]
    exemptions = spec.get("audit_exemptions", [])
    id_spec = spec.get("identifier_pseudonymization")
    if id_spec and not args.id_mapping:
        print(
            "identifier_pseudonymization is enabled; pass --id-mapping so that "
            "released identifiers use the same pseudonyms as results/final_main_tier_b",
            file=sys.stderr,
        )
        return 4
    known_ids = load_id_mapping(args.id_mapping) if id_spec else {}

    staged: dict[str, dict] = {}
    for rel in sorted(selected):
        source = repo_root / rel
        if source.is_symlink():
            raise SystemExit(f"unexpected symbolic link in allowlisted files: {rel}")
        raw = source.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise SystemExit(f"unexpected non-UTF-8 allowlisted file: {rel}")
        sanitized_text, applied = sanitize_text(text, sanitize_rules)
        exempt_rules = exemptions_for(rel, exemptions)
        data = sanitized_text.encode("utf-8")
        staged[rel] = {
            "data": data,
            "category": category_for(rel),
            "sanitized": bool(applied),
            "sanitize_rules": applied,
            "audit_exempt_rules": sorted(exempt_rules),
        }

    redaction_spec = spec.get("redact_json_fields")
    redaction_stats = {"files_redacted": 0, "fields_removed": 0}
    if redaction_spec:
        redaction_stats = apply_json_field_redactions(staged, redaction_spec)

    id_stats = {"files_with_fields_dropped": 0, "files_with_ids_replaced": 0}
    if id_spec:
        id_stats = apply_identifier_pseudonymization(staged, id_spec, known_ids)

    violations: list[dict] = []
    for rel, item in staged.items():
        data = item["data"]
        item["sha256"] = hashlib.sha256(data).hexdigest()
        item["bytes"] = len(data)
        violations.extend(
            audit_text(rel, data.decode("utf-8"), forbidden, set(item["audit_exempt_rules"]))
        )

    if violations:
        print("PUBLIC-DATA AUDIT FAILED; nothing copied.", file=sys.stderr)
        for item in violations:
            print(
                f"  {item['path']}:{item['line']}: {item['rule']} "
                "(matched content redacted)",
                file=sys.stderr,
            )
        print(f"{len(violations)} violation(s).", file=sys.stderr)
        return 2

    total_bytes = sum(item["bytes"] for item in staged.values())
    sanitized_count = sum(1 for item in staged.values() if item["sanitized"])
    print(f"allowlist: {allowlist_path}")
    print(f"files selected: {len(staged)}")
    print(f"total bytes: {total_bytes}")
    print(f"sanitized files: {sanitized_count}")
    if redaction_spec:
        print(f"JSON field redaction: {redaction_stats}")
    if id_spec:
        print(f"identifier pseudonymization: {id_stats}")

    if args.dry_run:
        print("dry-run: no output written")
        return 0

    if output.exists():
        if not args.force:
            print(f"output exists (use --force): {output}", file=sys.stderr)
            return 3
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for rel, item in sorted(staged.items()):
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(item["data"])

    manifest_lines = [f"{item['sha256']}  {rel}" for rel, item in sorted(staged.items())]
    (output / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    file_manifest = {
        "release_name": spec["release_name"],
        "allowlist_schema_version": spec["schema_version"],
        "file_count": len(staged),
        "total_bytes": total_bytes,
        "sanitized_file_count": sanitized_count,
        "redacted_field_file_count": redaction_stats["files_redacted"],
        "redacted_field_count": redaction_stats["fields_removed"],
        "files": [
            {
                "path": rel,
                "sha256": item["sha256"],
                "bytes": item["bytes"],
                "category": item["category"],
                "sanitized": item["sanitized"],
                "sanitize_rules": item["sanitize_rules"],
                "audit_exempt_rules": item["audit_exempt_rules"],
            }
            for rel, item in sorted(staged.items())
        ],
    }
    (output / "FILE_MANIFEST.json").write_text(
        json.dumps(file_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    csv_lines = ["path,sha256,bytes,category,sanitized"]
    for rel, item in sorted(staged.items()):
        csv_lines.append(
            f"{rel},{item['sha256']},{item['bytes']},{item['category']},{str(item['sanitized']).lower()}"
        )
    (output / "FILE_MANIFEST.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    print(f"candidate written: {output}")
    print(f"manifest: {output / 'MANIFEST.sha256'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
