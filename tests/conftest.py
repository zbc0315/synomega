"""Shared fixtures: a deterministic fake single-step model.

The neural backend needs a checkpoint and torch, so the search/route/metric
tests drive a hand-written model instead. That keeps the search logic under test
independent of model quality.
"""

from __future__ import annotations

import pytest

from synomega.singlestep.base import Prediction, SingleStepModel
from synomega.stock import InMemoryStock

# A toy "reaction network". Keys and values are canonical SMILES.
#   ester  <- acid + alcohol
#   amide  <- acid + amine
TOY_RULES: dict[str, list[tuple[tuple[str, ...], float]]] = {
    # CC(=O)OCC (ethyl acetate) <- acetic acid + ethanol
    "CCOC(C)=O": [(("CCO", "CC(=O)O"), 0.7), (("CCOC(C)=O",), 0.05)],
    # CC(=O)Nc1ccccc1 (acetanilide) <- acetic acid + aniline
    "CC(=O)Nc1ccccc1": [(("CC(=O)O", "Nc1ccccc1"), 0.8)],
    # a two-step target: make aniline from nitrobenzene
    "Nc1ccccc1": [(("O=[N+]([O-])c1ccccc1",), 0.6)],
}


class FakeModel(SingleStepModel):
    """Lookup-table model with deterministic scores."""

    name = "fake"

    def __init__(self, rules: dict | None = None):
        self.rules = rules if rules is not None else TOY_RULES
        self.calls = 0

    def predict(self, smiles: str, top_k: int = 50) -> list[Prediction]:
        self.calls += 1
        out = [
            Prediction(reactants=tuple(sorted(r)), score=s)
            for r, s in self.rules.get(smiles, [])
        ]
        out.sort(key=lambda p: -p.score)
        return out[:top_k]


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def stock() -> InMemoryStock:
    """Acetic acid, ethanol, nitrobenzene are purchasable; aniline is not."""
    return InMemoryStock.from_smiles(
        ["CC(=O)O", "CCO", "O=[N+]([O-])c1ccccc1"], name="toy"
    )
