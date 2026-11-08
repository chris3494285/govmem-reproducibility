"""Pytest configuration separating public offline tests from private ones.

Some tests exercise pipelines whose inputs are intentionally excluded from the
public release: licensed benchmark text (LongMemEval, Memora, OAKS Novel) and
private/local artifacts under ``results/**/private/``, developer-only manifests,
historical analysis outputs, and the external manuscript. Those tests are
enumerated in ``tests/public_release_test_scope.json`` and the matching pytest
marker is applied to each here.

By default marked tests are **skipped** so that ``pytest`` runs the public
offline suite that executes with no private data and no paid API calls. Pass
``--include-private`` to run everything (for example on the developer machine
that has the local datasets and private artifacts).

This mechanism changes only test selection. It never fabricates fixtures and
never presents synthetic data as an experimental result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

SCOPE_PATH = Path(__file__).with_name("public_release_test_scope.json")
_RELEASE_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_MARKERS = (
    "requires_private_artifacts",
    "requires_licensed_data",
    "requires_source_checkout",
)
SKIP_REASON = (
    "requires private/local or licensed artifacts excluded from the public "
    "release; run with --include-private to execute it"
)

# A few test modules read a non-public artifact at *import* time. The
# aggregate-only public release omits every row-level result file, so those
# modules cannot even be imported inside the snapshot and pytest would abort
# collection. Ignore such a module only when its import-time input is absent;
# on a developer tree (or any checkout that still has the artifact) it is
# collected and executed exactly as before. This never fabricates data.
_IMPORT_TIME_ARTIFACT_MODULES = {
    "test_formal_300_validation.py": (
        _RELEASE_ROOT / "results/formal_300_validation_v1/execution_plan.jsonl"
    ),
}
_IMPORT_TIME_ENV_REQUIREMENTS = {
    "test_formal_300_validation.py": (
        "GOVMEM_FORMAL_BLOCK_FINGERPRINT",
        "GOVMEM_FORMAL_ANTHROPIC_ATTEMPT3_REQUEST_ID",
        "GOVMEM_FORMAL_OPENAI_NO_TEXT_REQUEST_ID",
        "GOVMEM_FORMAL_ATTEMPT4_SMOKE_REQUEST_ID",
    ),
}

collect_ignore = sorted(
    module
    for module, artifact in _IMPORT_TIME_ARTIFACT_MODULES.items()
    if not artifact.is_file()
    or any(
        not os.environ.get(name)
        for name in _IMPORT_TIME_ENV_REQUIREMENTS.get(module, ())
    )
)


def _load_scope() -> dict[str, dict[str, str]]:
    """Map each scoped test node id to the marker name it should receive."""
    if not SCOPE_PATH.is_file():
        return {}
    data = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    scope: dict[str, dict[str, str]] = {}
    for marker, entry in data.get("categories", {}).items():
        if marker not in PRIVATE_MARKERS:
            raise ValueError(f"unknown test-scope marker: {marker}")
        for node in entry.get("tests", []):
            scope[node] = {"marker": marker, "reason": entry.get("description", "")}
    return scope


_SCOPE = _load_scope()


def _node_id(item: pytest.Item) -> str:
    return item.nodeid.replace("\\", "/")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--include-private",
        action="store_true",
        default=False,
        help="also run tests marked requires_private_artifacts/requires_licensed_data",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    include_private = config.getoption("--include-private")
    for item in items:
        entry = _SCOPE.get(_node_id(item))
        if entry is None and not any(item.get_closest_marker(name) for name in PRIVATE_MARKERS):
            continue
        marker_name = entry["marker"] if entry else next(
            name for name in PRIVATE_MARKERS if item.get_closest_marker(name)
        )
        item.add_marker(getattr(pytest.mark, marker_name))
        if not include_private:
            item.add_marker(pytest.mark.skip(reason=SKIP_REASON))
