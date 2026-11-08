"""Public manuscript-count checks for the JSS aggregate package."""

from __future__ import annotations

from pathlib import Path

from scripts.audit_jss_publication import validate_claims, validate_oaks_inventory


def test_public_aggregates_match_manuscript_claims():
    root = Path(__file__).resolve().parents[1]
    claims = validate_claims(root)

    assert claims["base_examples"] == 731
    assert claims["supported_pairs"] == 2080
    assert claims["conditions"] == 4891
    assert claims["providers"] == 6
    assert claims["resolved_formal_triplets"] == 298
    assert claims["resolved_distinct_bases"] == 238
    assert claims["bootstrap_iterations"] == 10000
    assert claims["bootstrap_seed"] == 20261010


def test_license_and_citation_metadata_are_scoped():
    root = Path(__file__).resolve().parents[1]
    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    scope_text = (root / "LICENSE_SCOPE.md").read_text(encoding="utf-8")
    approval = (root / "LICENSE_APPROVAL.md").read_text(encoding="utf-8")
    materials = (root / "MATERIALS_REDISTRIBUTION_NOTICE.md").read_text(
        encoding="utf-8"
    )
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")

    assert license_text.startswith("MIT License\n")
    assert "does not apply to third-party" in license_text
    assert "Approval status" in scope_text
    assert "all four coauthors" in approval.lower()
    assert "no employer, university, funder, or contractor owns or restricts" in (
        approval.lower()
    )
    assert "does not place those\nmaterials under the code-only MIT grant" in materials
    assert 'title: "Stress-Testing the Control Plane of Agent Memory"' in citation
    for author in ("Liu", "Wang", "Shen", "Shang"):
        assert f"family-names: {author}" in citation
    assert "doi:" not in citation.lower()


def test_oaks_derived_artifact_inventory_is_exact():
    root = Path(__file__).resolve().parents[1]
    inventory = validate_oaks_inventory(root)

    assert inventory["observed"] == 51
    assert inventory["documented"] == 51
    assert inventory["missing"] == []
    assert inventory["extra"] == []
