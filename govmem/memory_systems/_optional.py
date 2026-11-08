"""Small helpers shared by optional memory-system adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from .base import MemorySystemDependencyError


def require_dependency(import_name: str, install_name: str, system_name: str) -> Any:
    try:
        module = __import__(import_name, fromlist=["*"])
    except ImportError as exc:
        raise MemorySystemDependencyError(
            f"{system_name} adapter requires optional dependency {install_name!r}. "
            f"Install it in the experiment environment (for example: pip install {install_name}) "
            "or inject an initialized backend for tests."
        ) from exc
    return module


def call_backend(backend: Any, names: tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
    for name in names:
        method = getattr(backend, name, None)
        if callable(method):
            try:
                return method(*args, **kwargs)
            except TypeError:
                if kwargs:
                    return method(*args)
                raise
    raise AttributeError(f"Backend {type(backend).__name__} has none of methods {names}")


def metadata_dict(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata) if metadata is not None else {}
