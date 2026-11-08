"""Provider-neutral reconstruction engine using composition.

The established implementation is retained as a compatibility component so
the frozen OpenAI path keeps byte-compatible prompts, chunking, and validation.
The public engine itself has no provider-specific inheritance relationship.
"""

from typing import Any

from .openai_reconstruction import OpenAIGovernanceStateReconstructor
from .reconstruction_provider import ReconstructionProvider


class GovernanceStateReconstructionEngine:
    """Provider-injectable reconstruction engine with shared semantics.

    Composition is intentional: the provider is a transport dependency, while
    the existing implementation remains the single source of reconstruction
    policy and semantic validation during this compatibility refactor.
    """

    def __init__(self, *, provider: ReconstructionProvider | None = None, **kwargs: Any) -> None:
        object.__setattr__(self, "_implementation", OpenAIGovernanceStateReconstructor(provider=provider, **kwargs))

    def reconstruct(self, input_data):
        return self._implementation.reconstruct(input_data)

    def __getattr__(self, name: str) -> Any:
        # Preserve access to the existing telemetry/configuration surface for
        # callers while keeping the engine independent of provider inheritance.
        return getattr(self._implementation, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name != "_implementation" and "_implementation" in self.__dict__:
            setattr(self._implementation, name, value)
            return
        object.__setattr__(self, name, value)


__all__ = ["GovernanceStateReconstructionEngine"]
