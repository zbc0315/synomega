"""Single-step retrosynthesis interface.

Everything above this layer (search, synthesizability) talks to a
`SingleStepModel` and nothing else. A backend only has to turn a product SMILES
into scored reactant sets; whether it does so with a GNN, a transformer, or
plain template matching is invisible to the planner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Prediction:
    """One candidate disconnection of a product."""

    reactants: tuple[str, ...]        # canonical SMILES, sorted
    score: float                     # higher is better; probability-like
    template_id: int | None = None
    meta: dict = field(default_factory=dict, compare=False)

    @property
    def smiles(self) -> str:
        return ".".join(self.reactants)

    def __repr__(self) -> str:
        return f"Prediction({self.smiles!r}, score={self.score:.4f})"


class SingleStepModel(ABC):
    """Product SMILES -> ranked reactant candidates."""

    #: Human-readable backend name, used in logs and result metadata.
    name: str = "single-step"

    @abstractmethod
    def predict(self, smiles: str, top_k: int = 50) -> list[Prediction]:
        """Return up to `top_k` candidate reactant sets, best first."""

    def predict_batch(
        self, smiles: list[str], top_k: int = 50
    ) -> list[list[Prediction]]:
        """Batched variant.

        The default loops over `predict`. GPU-backed models **must** override
        this — search spends most of its wall-clock here, and one-molecule-at-a-
        time inference leaves the accelerator idle.
        """
        return [self.predict(s, top_k) for s in smiles]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


__all__ = ["Prediction", "SingleStepModel"]
