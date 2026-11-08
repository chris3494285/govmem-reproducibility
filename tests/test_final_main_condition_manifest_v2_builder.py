import hashlib
from pathlib import Path

import pytest

from scripts.build_final_main_condition_manifest_v2 import build


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA = "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"


def test_production_builder_exists_and_has_frozen_inputs():
    script = ROOT / "scripts/build_final_main_condition_manifest_v2.py"
    assert script.is_file()
    assert "build_final_main_governance_condition_manifest_v2" in script.read_text()


def test_builder_refuses_overwrite_without_force(tmp_path):
    output = tmp_path / "condition.jsonl"
    output.write_text("sentinel\n")
    with pytest.raises(FileExistsError):
        build(
            output=output,
            summary_path=tmp_path / "summary.json",
            freeze_path=tmp_path / "freeze.json",
        )
    assert output.read_text() == "sentinel\n"


def test_two_temporary_builds_are_byte_identical(tmp_path):
    paths = []
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        output = directory / "condition.jsonl"
        build(
            output=output,
            summary_path=directory / "summary.json",
            freeze_path=directory / "freeze.json",
        )
        paths.append(output)

    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert hashlib.sha256(paths[0].read_bytes()).hexdigest() == EXPECTED_SHA
