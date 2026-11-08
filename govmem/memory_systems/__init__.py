"""Optional external memory-system validation infrastructure.

The adapters in this package are intentionally separate from GovMem's frozen
governance and scoring code.  They normalize native retrieval payloads for
later validation runs; importing this package does not import any optional
third-party memory system.
"""

from .base import (
    MemoryRetrieval,
    MemoryRetrievalResult,
    MemorySystemAdapter,
    MemorySystemDependencyError,
    normalize_retrieval,
)
from .amem import AMemAdapter
from .langmem import LangMemAdapter
from .mem0 import Mem0Adapter

__all__ = [
    "MemoryRetrieval",
    "MemoryRetrievalResult",
    "MemorySystemAdapter",
    "MemorySystemDependencyError",
    "normalize_retrieval",
    "Mem0Adapter",
    "LangMemAdapter",
    "AMemAdapter",
]
