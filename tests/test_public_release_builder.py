"""Security regression tests for the deny-by-default release builder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _allowlist(include: list[str]) -> dict:
    return {
        "schema_version": "test-public-release-allowlist-v1",
        "release_name": "builder-test",
        "include": include,
        "exclude": [],
        "sanitize_rules": [],
        "redact_json_fields": {
            "drop_json_fields": ["private_identifier", "private_path"]
        },
        "forbidden_patterns": [
            {
                "name": "block_or_slot_fingerprint",
                "regex": "(pilot-block|formal-300-slot)::[0-9a-f]{16,}",
                "extensions": ["*"],
            },
            {
                "name": "absolute_user_path",
                "regex": "/Users/[A-Za-z]|/home/[A-Za-z]",
                "extensions": ["*"],
            },
        ],
        "audit_exemptions": [],
    }


def _run_builder(repo: Path, allowlist: Path, output: Path) -> subprocess.CompletedProcess:
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_public_release_candidate.py"
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo),
            "--allowlist",
            str(allowlist),
            "--output",
            str(output),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )


def test_builder_redacts_reviewed_fields_and_preserves_audits(tmp_path: Path) -> None:
    synthetic_id = "pilot-" + "block::0123456789abcdef"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "public.txt").write_text("aggregate release\n", encoding="utf-8")
    (repo / "aggregate.json").write_text(
        json.dumps(
            {
                "resolved": 298,
                "private_identifier": synthetic_id,
                "nested": {"private_path": "results/private/example.jsonl"},
            }
        ),
        encoding="utf-8",
    )
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(_allowlist(["public.txt", "aggregate.json"])),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    result = _run_builder(repo, allowlist, output)

    assert result.returncode == 0, result.stderr
    assert (output / "public.txt").read_text(encoding="utf-8") == "aggregate release\n"
    aggregate = json.loads((output / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate == {"resolved": 298, "nested": {}}
    manifest = json.loads((output / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["redacted_field_file_count"] == 1
    assert "private_identifier" not in (output / "aggregate.json").read_text(encoding="utf-8")


def test_builder_fails_closed_without_echoing_restricted_match(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    restricted = "pilot-" + "block::0123456789abcdef0123456789abcdef"
    (repo / "restricted.txt").write_text(restricted + "\n", encoding="utf-8")
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(json.dumps(_allowlist(["restricted.txt"])), encoding="utf-8")
    output = tmp_path / "must-not-exist"

    result = _run_builder(repo, allowlist, output)

    assert result.returncode == 2
    assert not output.exists()
    assert "PUBLIC-DATA AUDIT FAILED" in result.stderr
    assert "block_or_slot_fingerprint" in result.stderr
    assert "matched content redacted" in result.stderr
    assert restricted not in result.stderr
