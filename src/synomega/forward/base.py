"""Forward reaction prediction interface.

Where a `SingleStepModel` turns a product into ranked reactant sets, a
`ForwardModel` turns a reactant set into ranked product candidates. It is a
separate base class: a forward model is not a `SingleStepModel`, so the planner
does not consume it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ForwardPrediction:
    """One predicted product of a reaction."""

    product: str                      # canonical SMILES
    score: float                      # higher is better; probability-like
    template_id: int | None = None
    meta: dict = field(default_factory=dict, compare=False)

    @property
    def smiles(self) -> str:
        return self.product

    def __repr__(self) -> str:
        return f"ForwardPrediction({self.product!r}, score={self.score:.4f})"


class ForwardModel(ABC):
    """Reactant SMILES -> ranked product candidates."""

    #: Human-readable backend name, used in logs and result metadata.
    name: str = "forward"

    @abstractmethod
    def predict(self, reactants: str, top_k: int = 10) -> list[ForwardPrediction]:
        """Return up to `top_k` candidate products, best first.

        `reactants` is a single SMILES string; multiple reactants are
        dot-separated (e.g. ``"CC(=O)O.CN"``).
        """

    def predict_batch(
        self, reactants: list[str], top_k: int = 10
    ) -> list[list[ForwardPrediction]]:
        """Batched variant. GPU-backed models should override this."""
        return [self.predict(r, top_k) for r in reactants]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


__all__ = ["ForwardPrediction", "ForwardModel"]
