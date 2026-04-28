"""turboquant.search -- Nearest-neighbor search with TurboQuant."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from turboquant.search.index import TurboQuantIndex

if TYPE_CHECKING:
    from turboquant.search.langchain import TurboQuantVectorStore

__all__ = ["TurboQuantIndex", "TurboQuantVectorStore"]


def __getattr__(name: str) -> Any:
    # Lazy import so importing turboquant.search doesn't pull in langchain-core
    # unless the optional integration is actually requested.
    if name == "TurboQuantVectorStore":
        from turboquant.search.langchain import TurboQuantVectorStore

        return TurboQuantVectorStore
    raise AttributeError(f"module 'turboquant.search' has no attribute {name!r}")
