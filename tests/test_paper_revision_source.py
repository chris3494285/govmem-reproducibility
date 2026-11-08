from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "paper_aiopen_final"
TEXT_GLOBS = ("*.tex", "*.md", "*.bib")
# Provenance documents intentionally quote the removed values.
PROVENANCE = {"REVISION_AUDIT.md", "youting_review.md", "13_README.md"}


def _source_text(include_provenance: bool = False) -> str:
    parts = []
    for pattern in TEXT_GLOBS:
        for path in sorted(SRC.glob(pattern)):
            if not include_provenance and path.name in PROVENANCE:
                continue
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_provenance_audit_lists_removed_values():
    audit = (SRC / "REVISION_AUDIT.md").read_text(encoding="utf-8")
    for removed in ("27.55", "27.51", "2.33", "4,160"):
        assert removed in audit, f"audit should record removal of {removed}"


def test_no_obsolete_behavioral_numbers_or_scorer_language():
    text = _source_text()
    for obsolete in ("27.55", "27.51", "2.33", "4,160", "2,002", "28.84", "28.79"):
        assert obsolete not in text, f"obsolete value survived: {obsolete}"
    lowered = text.lower()
    assert "deterministic task correctness" not in lowered
    assert "deterministic benchmark-native" not in lowered
    assert "benchmark-native scoring" not in lowered


def test_required_final_numbers_present_with_consistent_framing():
    text = _source_text()
    for required in ("35.54", "32.67", "2.86", "96.25", "298", "900", "898"):
        assert required in text, f"required final value missing: {required}"
    normalized = text.replace("{,}", ",")
    for required in ("4,891", "2,080", "12,480"):
        assert required in normalized, f"required reconciliation value missing: {required}"
    # 4,891 = 731 + 2,080 + 2,080; 12,480 = 2,080 x 6
    assert "4,891" in normalized and "2,080" in normalized and "731" in normalized
    assert re.search(r"2,080\s*(?:\\times|×|x)\s*6", normalized)
    # control-plane vs behavioral separation
    assert "control-plane" in text.lower()
    assert "restoration gap" in text.lower()
    # unresolved observations and the human adjudication are disclosed
    assert re.search(r"unresolved", text, re.I)
    assert re.search(r"human[- ]adjudicat", text, re.I)


def test_revision_directory_is_self_contained_for_latex():
    main = (SRC / "00_main.tex").read_text(encoding="utf-8")
    inputs = re.findall(r"\\input\{([^}]+)\}", main)
    for name in inputs:
        assert (SRC / f"{name}.tex").exists(), f"missing \\input target: {name}"
    # every table file referenced from results exists
    results = (SRC / "06_results.tex").read_text(encoding="utf-8")
    for name in re.findall(r"\\input\{([^}]+)\}", results):
        assert (SRC / f"{name}.tex").exists(), f"missing table input: {name}"
    assert (SRC / "10_references.bib").exists()
    assert (SRC / "figures" / "figure1_control_plane.pdf").exists()
    assert (SRC / "REVISION_AUDIT.md").exists()
    assert (SRC / "youting_review.md").exists()
